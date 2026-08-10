# Model checksums

SHA-256 manifests for the model files the subprojects here measure against.

Every reference file in this repo — logit-kld baselines, nano-glm's bit-exact
gates, the A/B quant studies — is only meaningful against known model bytes.
Nothing else pins them: a `.bin` records the model *path* and description, not
its contents.

## Verify a copy

```bash
cd /path/to/model/dir
shasum -a 256 -c /path/to/llama-sandbox/checksums/GLM-5.2-UD-Q6_K.sha256   # macOS/Linux
```

```powershell
# Windows
Get-ChildItem *.gguf | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash $_ -Algorithm SHA256).Hash.ToLower(), $_.Name
}
```

**Do this after every copy of a model, before producing any baseline from it.**

**A hash immediately after a copy may only be verifying the page cache.** The
copy writes through cache, so a read straight afterwards can return the bytes
you intended while the platter holds something else — which is exactly how the
2026-08-07 incident hid for a day. To check what actually landed, re-verify
after a reboot, or bypass the cache.

**And the cache can be wrong in the other direction too.** On 2026-08-10 a
buffered hash reported shard 02 BAD on a model whose disk copy was perfect: a
descriptor table had been spliced into *cached pages* of a read-only mapping.
The two failures look identical from a `Get-FileHash`, so when one disagrees
with the manifest, settle it before doing anything drastic:

```powershell
.\hash_nocache.ps1 -Path D:\llms\UD-Q6_K\GLM-5.2-UD-Q6_K-00002-of-00014.gguf
```

- matches the manifest → the platter is fine, the cache is poisoned. **Reboot.**
- differs → the file really is damaged. Restore it from the other machine, and
  diff before overwriting (below).

## Tools

| file | what |
|---|---|
| `chunk_hash.py` | per-chunk hashes, to localize a difference between two copies |
| `bindiff.py` | byte-level diff of a damaged file against a good one |
| `scan_splice.py` | finds the descriptor-splice signature in one file, **no reference copy needed** |
| `hash_nocache.ps1` | SHA-256 bypassing the Windows page cache (disk truth) |

`scan_splice.py` exists because the damage here has twice been a page-descriptor
list — 8-byte records `f1 04 XX XX 07 00 40 00` stepping 2 MiB per record —
rather than random bit rot. That is specific enough to identify on sight, which
means a suspect file can be diagnosed on its own, before the good copy is to
hand and before overwriting destroys the evidence. Run it on a known-good shard
too: a signature scanner that fires on healthy data is worthless.

## Why this exists

2026-08-07: a bit-exact gate that had passed for two days started failing by
~8.6e-04 mean KL — small enough to look like numerical noise, large enough to
break byte-comparison. Source, binaries, ggml commit, thread count, OpenMP
settings and model mtimes/sizes were all identical and all eliminated in turn.

The cause was a corrupt shard: `GLM-5.2-UD-Q6_K-00006-of-00014.gguf` on the
Windows machine's exFAT volume differed from the macOS copy. The file was the
right size, parsed as valid GGUF, loaded without complaint, and produced
plausible text — it just used slightly wrong weights for the experts in layers
26-32. A 626 GB copy had completed with no error and silently produced a
different file.

Two things made it hard to see, both worth remembering:

- **The page cache masked it for a day.** The copy wrote through cache, so runs
  on the day of the copy read the still-correct cached pages. Only after a
  reboot did anything read the bad bytes from disk. "It reproduced yesterday"
  is not evidence that the bytes on disk are good.
- **MoE routing made it position-dependent.** Corrupt expert weights only
  perturb tokens that route to those experts, so some positions stayed
  bit-exact while others drifted — a pattern that looks nothing like a
  compiler or threading difference, and which pointed at the real cause once
  noticed.

No filesystem in play here would have caught it: exFAT, NTFS and APFS all
checksum metadata at best, never file data.

## What the damage turned out to be

Diffing the corrupt files against good copies before overwriting them
(`bindiff.py`, reports in `shard06-corruption.txt` and
`shard14-corruption.txt`) settled the mechanism:

- shard 06: 680 bytes over 3 runs; shard 14: 1217 bytes over 13 runs
- the corrupt bytes are a **structured table** — repeating 8-byte records
  `f1 04 XX YY 07 00 40 00`, constant high dword, entries advancing with a
  **2 MiB stride** — i.e. an in-memory page-descriptor list, not noise
- most runs are **not sector-aligned** (`offset % 512` = 352, 439, 7, 80, …),
  so this happened above the storage layer, in a memory buffer

Conclusion: a Windows **writeback-DMA bug** transferred a descriptor structure
instead of the pages it described. Correct bytes in cache, wrong bytes on
disk — which is precisely why the cache masked it. ECC, WHEA, the storage
error log and `chkdsk` were all clean throughout, because nothing failed: the
wrong source address was handed down and faithfully written.

Fresh copies of both shards verified first time, so the write path is not
reproducibly broken. Re-running the gate with repaired weights reproduced the
pre-corruption results **bit-identically**, which confirms both the diagnosis
and that the earlier baselines were computed from good data.

The damage *shape* is what distinguishes a driver bug from failing hardware —
isolated bit flips would have meant the opposite conclusion. So diff a corrupt
file against a good copy **before** overwriting it; that evidence is
unrecoverable afterwards.
