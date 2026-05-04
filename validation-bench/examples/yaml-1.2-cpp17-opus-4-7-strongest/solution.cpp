
#include <bits/stdc++.h>
using namespace std;

static vector<int> S;
static int pos;
static int N;

static int peek(int off = 0) { int i = pos + off; return (i >= 0 && i < N) ? S[i] : -1; }
static int sav() { return pos; }
static void rst(int p) { pos = p; }
static bool eof() { return pos >= N; }
static bool atSOL() { return pos == 0 || (pos > 0 && (S[pos-1] == 0x0A || S[pos-1] == 0x0D)); }

static bool is_printable(int c) {
    return c == 0x09 || c == 0x0A || c == 0x0D ||
        (c >= 0x20 && c <= 0x7E) || c == 0x85 ||
        (c >= 0xA0 && c <= 0xD7FF) ||
        (c >= 0xE000 && c <= 0xFFFD) ||
        (c >= 0x10000 && c <= 0x10FFFF);
}
static bool is_bchar(int c) { return c == 0x0A || c == 0x0D; }
static bool is_nbchar(int c) { return is_printable(c) && !is_bchar(c) && c != 0xFEFF; }
static bool is_white(int c) { return c == 0x20 || c == 0x09; }
static bool is_nschar(int c) { return is_nbchar(c) && !is_white(c); }
static bool is_flowind(int c) { return c==','||c=='['||c==']'||c=='{'||c=='}'; }
static bool is_indicator(int c) {
    return c=='-'||c=='?'||c==':'||c==','||c=='['||c==']'||c=='{'||c=='}'||
           c=='#'||c=='&'||c=='*'||c=='!'||c=='|'||c=='>'||c=='\''||c=='"'||
           c=='%'||c=='@'||c=='`';
}
static bool is_decdig(int c) { return c >= '0' && c <= '9'; }
static bool is_hexdig(int c) { return is_decdig(c)||(c>='A'&&c<='F')||(c>='a'&&c<='f'); }
static bool is_letter(int c) { return (c>='A'&&c<='Z')||(c>='a'&&c<='z'); }
static bool is_wordchar(int c) { return is_decdig(c)||is_letter(c)||c=='-'; }

enum Ctx { BO, BI, BK, FO, FI, FK };
enum Chomp { CSTRIP, CCLIP, CKEEP };

static Ctx flow_in_ctx(Ctx c) { if (c == BK || c == FK) return FK; return FI; }

static bool at_forbidden() {
    if (!atSOL()) return false;
    if (peek() == '-' && peek(1) == '-' && peek(2) == '-') {
        int c4 = peek(3);
        if (c4 == -1 || is_white(c4) || is_bchar(c4)) return true;
    }
    if (peek() == '.' && peek(1) == '.' && peek(2) == '.') {
        int c4 = peek(3);
        if (c4 == -1 || is_white(c4) || is_bchar(c4)) return true;
    }
    return false;
}

static bool yaml_directive_seen = false;
static set<string> declared_handles;
static void reset_doc_state() { yaml_directive_seen = false; declared_handles.clear(); }

static bool s_separate(int n, Ctx c);
static bool s_l_comments();
static bool ns_flow_node(int n, Ctx c);
static bool ns_flow_yaml_node(int n, Ctx c);
static bool c_flow_json_node(int n, Ctx c);
static bool ns_flow_seq_entry(int n, Ctx c);
static bool ns_flow_map_entry(int n, Ctx c);
static bool ns_flow_pair(int n, Ctx c);
static bool s_l_block_node(int n, Ctx c);
static bool s_l_block_indented(int n, Ctx c);
static bool ns_l_compact_sequence(int n);
static bool ns_l_compact_mapping(int n);
static bool ns_l_block_map_entry(int n);
static bool c_l_block_seq_entry(int n);
static bool ns_flow_map_implicit_entry(int n, Ctx c);
static bool ns_flow_map_explicit_entry(int n, Ctx c);
static bool c_ns_flow_map_separate_value(int n, Ctx c);
static bool c_ns_flow_map_adjacent_value(int n, Ctx c);
static bool ns_s_implicit_yaml_key(Ctx c);
static bool c_s_implicit_json_key(Ctx c);
static bool c_flow_sequence(int n, Ctx c);
static bool c_flow_mapping(int n, Ctx c);
static bool ns_flow_yaml_content(int n, Ctx c);
static bool c_flow_json_content(int n, Ctx c);
static bool ns_flow_content(int n, Ctx c);
static bool c_ns_properties(int n, Ctx c);
static bool c_ns_anchor_property();
static bool ns_anchor_name();
static bool c_ns_tag_property();
static bool seq_space(int n, Ctx c);
static bool l_block_mapping(int n);

static bool b_break() {
    if (peek() == 0x0D) { pos++; if (peek() == 0x0A) pos++; return true; }
    if (peek() == 0x0A) { pos++; return true; }
    return false;
}
static bool b_non_content() { return b_break(); }
static bool b_as_line_feed() { return b_break(); }
static bool b_as_space() { return b_break(); }

static bool s_indent(int n) {
    if (n <= 0) return true;
    int sp = sav();
    for (int i = 0; i < n; i++) {
        if (peek() != 0x20) { rst(sp); return false; }
        pos++;
    }
    return true;
}
static void s_indent_lt(int n) { int c = 0; while (c < n - 1 && peek() == 0x20) { pos++; c++; } }
static void s_indent_le(int n) { int c = 0; while (c < n && peek() == 0x20) { pos++; c++; } }

static bool s_separate_in_line() {
    if (is_white(peek())) { while (is_white(peek())) pos++; return true; }
    if (atSOL()) return true;
    return false;
}

static bool s_block_line_prefix(int n) { return s_indent(n); }
static bool s_flow_line_prefix(int n) {
    if (!s_indent(n)) return false;
    int sp = sav();
    if (!s_separate_in_line()) rst(sp);
    return true;
}
static bool s_line_prefix(int n, Ctx c) {
    if (c == BO || c == BI) return s_block_line_prefix(n);
    return s_flow_line_prefix(n);
}

static bool l_empty(int n, Ctx c) {
    int sp = sav();
    int sp2 = sav();
    if (!s_line_prefix(n, c)) {
        rst(sp2);
        s_indent_lt(n);
    }
    if (!b_as_line_feed()) { rst(sp); return false; }
    return true;
}
static bool b_l_trimmed(int n, Ctx c) {
    int sp = sav();
    if (!b_non_content()) { rst(sp); return false; }
    if (!l_empty(n, c)) { rst(sp); return false; }
    while (true) { int sp2 = sav(); if (!l_empty(n, c)) { rst(sp2); break; } }
    return true;
}
static bool b_l_folded(int n, Ctx c) {
    int sp = sav();
    if (b_l_trimmed(n, c)) return true;
    rst(sp);
    return b_as_space();
}
static bool s_flow_folded(int n) {
    int sp = sav();
    int sp2 = sav();
    if (!s_separate_in_line()) rst(sp2);
    if (!b_l_folded(n, FI)) { rst(sp); return false; }
    if (at_forbidden()) { rst(sp); return false; }
    if (!s_flow_line_prefix(n)) { rst(sp); return false; }
    return true;
}

static bool c_nb_comment_text() {
    if (peek() != '#') return false;
    pos++;
    while (is_nbchar(peek())) pos++;
    return true;
}
static bool b_comment() { if (eof()) return true; return b_non_content(); }
static bool s_b_comment() {
    int sp = sav();
    int sp2 = sav();
    if (s_separate_in_line()) {
        int sp3 = sav();
        if (!c_nb_comment_text()) rst(sp3);
    } else { rst(sp2); }
    if (!b_comment()) { rst(sp); return false; }
    return true;
}
static bool l_comment() {
    int sp = sav();
    if (!s_separate_in_line()) { rst(sp); return false; }
    int sp2 = sav();
    if (!c_nb_comment_text()) rst(sp2);
    if (!b_comment()) { rst(sp); return false; }
    if (sp == sav()) return false;
    return true;
}
static bool s_l_comments() {
    int sp = sav();
    if (!atSOL()) { if (!s_b_comment()) { rst(sp); return false; } }
    while (true) { int sp2 = sav(); if (!l_comment()) { rst(sp2); break; } }
    return true;
}

static bool s_separate_lines(int n) {
    int sp = sav();
    if (s_l_comments() && s_flow_line_prefix(n)) return true;
    rst(sp);
    return s_separate_in_line();
}
static bool s_separate(int n, Ctx c) {
    if (c == BK || c == FK) return s_separate_in_line();
    return s_separate_lines(n);
}

static bool is_uri_char_one() {
    int c = peek();
    if (c == '%') {
        if (is_hexdig(peek(1)) && is_hexdig(peek(2))) { pos += 3; return true; }
        return false;
    }
    if (is_wordchar(c) || c=='#'||c==';'||c=='/'||c=='?'||c==':'||c=='@'||
        c=='&'||c=='='||c=='+'||c=='$'||c==','||c=='_'||c=='.'||c=='!'||
        c=='~'||c=='*'||c=='\''||c=='('||c==')'||c=='['||c==']') {
        pos++; return true;
    }
    return false;
}
static bool is_tag_char_one() {
    int c = peek();
    if (c == '!' || is_flowind(c)) return false;
    return is_uri_char_one();
}

static bool ns_char_one() { if (is_nschar(peek())) { pos++; return true; } return false; }
static bool ns_directive_name() { if (!ns_char_one()) return false; while (ns_char_one()) {} return true; }
static bool ns_directive_parameter() { return ns_directive_name(); }
static bool ns_yaml_version() {
    if (!is_decdig(peek())) return false;
    while (is_decdig(peek())) pos++;
    if (peek() != '.') return false;
    pos++;
    if (!is_decdig(peek())) return false;
    while (is_decdig(peek())) pos++;
    return true;
}
static bool ns_yaml_directive() {
    int sp = sav();
    if (peek() != 'Y' || peek(1) != 'A' || peek(2) != 'M' || peek(3) != 'L') return false;
    pos += 4;
    if (!s_separate_in_line()) { rst(sp); return false; }
    if (!ns_yaml_version()) { rst(sp); return false; }
    if (yaml_directive_seen) { rst(sp); return false; }
    yaml_directive_seen = true;
    return true;
}
static bool c_primary_tag_handle() { if (peek() != '!') return false; pos++; return true; }
static bool c_secondary_tag_handle() { if (peek() != '!' || peek(1) != '!') return false; pos += 2; return true; }
static bool c_named_tag_handle() {
    int sp = sav();
    if (peek() != '!') return false;
    pos++;
    if (!is_wordchar(peek())) { rst(sp); return false; }
    while (is_wordchar(peek())) pos++;
    if (peek() != '!') { rst(sp); return false; }
    pos++;
    return true;
}
static bool c_tag_handle() {
    int sp = sav();
    if (c_named_tag_handle()) return true; rst(sp);
    if (c_secondary_tag_handle()) return true; rst(sp);
    if (c_primary_tag_handle()) return true; rst(sp);
    return false;
}
static bool c_ns_local_tag_prefix() {
    if (peek() != '!') return false;
    pos++;
    while (is_uri_char_one()) {}
    return true;
}
static bool ns_global_tag_prefix() {
    if (!is_tag_char_one()) return false;
    while (is_uri_char_one()) {}
    return true;
}
static bool ns_tag_prefix() {
    int sp = sav();
    if (c_ns_local_tag_prefix()) return true; rst(sp);
    return ns_global_tag_prefix();
}
static bool ns_tag_directive() {
    int sp = sav();
    if (peek() != 'T' || peek(1) != 'A' || peek(2) != 'G') return false;
    pos += 3;
    if (!s_separate_in_line()) { rst(sp); return false; }
    int hs = sav();
    if (!c_tag_handle()) { rst(sp); return false; }
    int he = sav();
    if (!s_separate_in_line()) { rst(sp); return false; }
    if (!ns_tag_prefix()) { rst(sp); return false; }
    string handle;
    for (int i = hs; i < he; i++) handle += (char)(S[i] & 0xFF);
    if (declared_handles.count(handle)) { rst(sp); return false; }
    declared_handles.insert(handle);
    return true;
}
static bool ns_reserved_directive() {
    int sp = sav();
    if (!ns_directive_name()) return false;
    string name;
    for (int i = sp; i < pos; i++) name += (char)(S[i] & 0xFF);
    if (name == "YAML" || name == "TAG") { rst(sp); return false; }
    while (true) {
        int sp2 = sav();
        if (!s_separate_in_line()) { rst(sp2); break; }
        if (!ns_directive_parameter()) { rst(sp2); break; }
    }
    return true;
}
static bool l_directive() {
    int sp = sav();
    if (peek() != '%') return false;
    pos++;
    int sp2 = sav();
    if (!ns_yaml_directive()) {
        rst(sp2);
        if (!ns_tag_directive()) {
            rst(sp2);
            if (!ns_reserved_directive()) { rst(sp); return false; }
        }
    }
    if (!s_l_comments()) { rst(sp); return false; }
    return true;
}

static bool ns_anchor_char_one() {
    int c = peek();
    if (is_nschar(c) && !is_flowind(c)) { pos++; return true; }
    return false;
}
static bool ns_anchor_name() {
    if (!ns_anchor_char_one()) return false;
    while (ns_anchor_char_one()) {}
    return true;
}
static bool c_ns_anchor_property() {
    int sp = sav();
    if (peek() != '&') return false;
    pos++;
    if (!ns_anchor_name()) { rst(sp); return false; }
    return true;
}
static bool c_verbatim_tag() {
    int sp = sav();
    if (peek() != '!' || peek(1) != '<') return false;
    pos += 2;
    if (!is_uri_char_one()) { rst(sp); return false; }
    while (is_uri_char_one()) {}
    if (peek() != '>') { rst(sp); return false; }
    pos++;
    return true;
}
static bool c_ns_shorthand_tag() {
    int sp = sav();
    int hs = sav();
    if (!c_tag_handle()) return false;
    int he = sav();
    if (!is_tag_char_one()) { rst(sp); return false; }
    while (is_tag_char_one()) {}
    string handle;
    for (int i = hs; i < he; i++) handle += (char)(S[i] & 0xFF);
    if (handle != "!" && handle != "!!" && !declared_handles.count(handle)) {
        rst(sp); return false;
    }
    return true;
}
static bool c_non_specific_tag() { if (peek() != '!') return false; pos++; return true; }
static bool c_ns_tag_property() {
    int sp = sav();
    if (c_verbatim_tag()) return true; rst(sp);
    if (c_ns_shorthand_tag()) return true; rst(sp);
    return c_non_specific_tag();
}
static bool c_ns_properties(int n, Ctx c) {
    int sp = sav();
    if (c_ns_tag_property()) {
        int sp2 = sav();
        if (s_separate(n, c) && c_ns_anchor_property()) return true;
        rst(sp2);
        return true;
    }
    rst(sp);
    if (c_ns_anchor_property()) {
        int sp2 = sav();
        if (s_separate(n, c) && c_ns_tag_property()) return true;
        rst(sp2);
        return true;
    }
    return false;
}

static bool c_ns_esc_char() {
    int sp = sav();
    if (peek() != '\\') return false;
    pos++;
    int c = peek();
    if (c == '0' || c == 'a' || c == 'b' || c == 't' || c == 0x09 ||
        c == 'n' || c == 'v' || c == 'f' || c == 'r' || c == 'e' ||
        c == 0x20 || c == '"' || c == '/' || c == '\\' || c == 'N' ||
        c == '_' || c == 'L' || c == 'P') { pos++; return true; }
    if (c == 'x') {
        pos++;
        for (int i = 0; i < 2; i++) { if (!is_hexdig(peek())) { rst(sp); return false; } pos++; }
        return true;
    }
    if (c == 'u') {
        pos++;
        for (int i = 0; i < 4; i++) { if (!is_hexdig(peek())) { rst(sp); return false; } pos++; }
        return true;
    }
    if (c == 'U') {
        pos++;
        for (int i = 0; i < 8; i++) { if (!is_hexdig(peek())) { rst(sp); return false; } pos++; }
        return true;
    }
    rst(sp); return false;
}

static bool nb_double_char() {
    int sp = sav();
    if (c_ns_esc_char()) return true;
    rst(sp);
    int c = peek();
    if (c == '\\' || c == '"') return false;
    if (c == 0x09 || (c >= 0x20 && c <= 0x10FFFF)) { pos++; return true; }
    return false;
}
static bool ns_double_char() {
    int sp = sav();
    if (!nb_double_char()) return false;
    int c = S[sp];
    if (c == ' ' || c == '\t') { rst(sp); return false; }
    return true;
}
static bool nb_double_one_line() { while (nb_double_char()) {} return true; }
static bool s_double_escaped(int n) {
    int sp = sav();
    while (peek() == ' ' || peek() == '\t') pos++;
    if (peek() != '\\') { rst(sp); return false; }
    pos++;
    if (!b_non_content()) { rst(sp); return false; }
    while (true) {
        int sp2 = sav();
        if (at_forbidden()) { rst(sp2); break; }
        if (!l_empty(n, FI)) { rst(sp2); break; }
    }
    if (at_forbidden()) { rst(sp); return false; }
    if (!s_flow_line_prefix(n)) { rst(sp); return false; }
    return true;
}
static bool s_double_break(int n) {
    int sp = sav();
    if (s_double_escaped(n)) return true;
    rst(sp);
    return s_flow_folded(n);
}
static bool nb_ns_double_in_line() {
    while (true) {
        int sp = sav();
        while (peek() == ' ' || peek() == '\t') pos++;
        if (!ns_double_char()) { rst(sp); break; }
    }
    return true;
}
static bool s_double_next_line(int n) {
    int sp = sav();
    if (!s_double_break(n)) return false;
    int sp2 = sav();
    if (ns_double_char()) {
        nb_ns_double_in_line();
        int sp3 = sav();
        if (!s_double_next_line(n)) {
            rst(sp3);
            while (peek() == ' ' || peek() == '\t') pos++;
        }
    } else { rst(sp2); }
    return true;
}
static bool nb_double_multi_line(int n) {
    nb_ns_double_in_line();
    int sp = sav();
    if (!s_double_next_line(n)) {
        rst(sp);
        while (peek() == ' ' || peek() == '\t') pos++;
    }
    return true;
}
static bool nb_double_text(int n, Ctx c) {
    if (c == FO || c == FI) return nb_double_multi_line(n);
    return nb_double_one_line();
}
static bool c_double_quoted(int n, Ctx c) {
    int sp = sav();
    if (peek() != '"') return false;
    pos++;
    if (!nb_double_text(n, c)) { rst(sp); return false; }
    if (peek() != '"') { rst(sp); return false; }
    pos++;
    return true;
}

static bool nb_single_char() {
    if (peek() == '\'' && peek(1) == '\'') { pos += 2; return true; }
    int c = peek();
    if (c == '\'') return false;
    if (c == 0x09 || (c >= 0x20 && c <= 0x10FFFF)) { pos++; return true; }
    return false;
}
static bool ns_single_char() {
    int sp = sav();
    if (!nb_single_char()) return false;
    int c = S[sp];
    if (c == ' ' || c == '\t') { rst(sp); return false; }
    return true;
}
static bool nb_single_one_line() { while (nb_single_char()) {} return true; }
static bool nb_ns_single_in_line() {
    while (true) {
        int sp = sav();
        while (peek() == ' ' || peek() == '\t') pos++;
        if (!ns_single_char()) { rst(sp); break; }
    }
    return true;
}
static bool s_single_next_line(int n) {
    int sp = sav();
    if (!s_flow_folded(n)) return false;
    int sp2 = sav();
    if (ns_single_char()) {
        nb_ns_single_in_line();
        int sp3 = sav();
        if (!s_single_next_line(n)) {
            rst(sp3);
            while (peek() == ' ' || peek() == '\t') pos++;
        }
    } else { rst(sp2); }
    return true;
}
static bool nb_single_multi_line(int n) {
    nb_ns_single_in_line();
    int sp = sav();
    if (!s_single_next_line(n)) {
        rst(sp);
        while (peek() == ' ' || peek() == '\t') pos++;
    }
    return true;
}
static bool nb_single_text(int n, Ctx c) {
    if (c == FO || c == FI) return nb_single_multi_line(n);
    return nb_single_one_line();
}
static bool c_single_quoted(int n, Ctx c) {
    int sp = sav();
    if (peek() != '\'') return false;
    pos++;
    if (!nb_single_text(n, c)) { rst(sp); return false; }
    if (peek() != '\'') { rst(sp); return false; }
    pos++;
    return true;
}

static bool ns_plain_first(Ctx c) {
    if (atSOL() && at_forbidden()) return false;
    int sp = sav();
    int ch = peek();
    if (is_nschar(ch) && !is_indicator(ch)) {
        if ((c == FI || c == FK) && is_flowind(ch)) return false;
        pos++; return true;
    }
    if (ch == '?' || ch == ':' || ch == '-') {
        pos++;
        int ch2 = peek();
        bool ok = is_nschar(ch2);
        if (ok && (c == FI || c == FK) && is_flowind(ch2)) ok = false;
        if (!ok) { rst(sp); return false; }
        return true;
    }
    return false;
}
static bool ns_plain_safe_char(Ctx c) {
    int ch = peek();
    if (!is_nschar(ch)) return false;
    if ((c == FI || c == FK) && is_flowind(ch)) return false;
    return true;
}
static bool ns_plain_char(Ctx c) {
    int sp = sav();
    int ch = peek();
    if (ch == '#') {
        if (sp > 0 && is_nschar(S[sp-1])) { pos++; return true; }
        return false;
    }
    if (ch == ':') {
        pos++;
        int ch2 = peek();
        bool ok = is_nschar(ch2);
        if (ok && (c == FI || c == FK) && is_flowind(ch2)) ok = false;
        if (!ok) { rst(sp); return false; }
        return true;
    }
    if (ns_plain_safe_char(c)) { pos++; return true; }
    return false;
}
static bool nb_ns_plain_in_line(Ctx c) {
    while (true) {
        int sp = sav();
        while (peek() == ' ' || peek() == '\t') pos++;
        if (!ns_plain_char(c)) { rst(sp); break; }
    }
    return true;
}
static bool ns_plain_one_line(Ctx c) {
    if (!ns_plain_first(c)) return false;
    nb_ns_plain_in_line(c);
    return true;
}
static bool s_ns_plain_next_line(int n, Ctx c) {
    int sp = sav();
    if (!s_flow_folded(n)) return false;
    if (at_forbidden()) { rst(sp); return false; }
    if (!ns_plain_char(c)) { rst(sp); return false; }
    nb_ns_plain_in_line(c);
    return true;
}
static bool ns_plain_multi_line(int n, Ctx c) {
    if (!ns_plain_one_line(c)) return false;
    while (true) { int sp = sav(); if (!s_ns_plain_next_line(n, c)) { rst(sp); break; } }
    return true;
}
static bool ns_plain(int n, Ctx c) {
    if (c == FO || c == FI) return ns_plain_multi_line(n, c);
    return ns_plain_one_line(c);
}

static bool ns_flow_yaml_content(int n, Ctx c) { return ns_plain(n, c); }
static bool c_flow_json_content(int n, Ctx c) {
    int sp = sav();
    if (c_flow_sequence(n, c)) return true; rst(sp);
    if (c_flow_mapping(n, c)) return true; rst(sp);
    if (c_single_quoted(n, c)) return true; rst(sp);
    if (c_double_quoted(n, c)) return true; rst(sp);
    return false;
}
static bool ns_flow_content(int n, Ctx c) {
    int sp = sav();
    if (ns_flow_yaml_content(n, c)) return true; rst(sp);
    return c_flow_json_content(n, c);
}
static bool c_ns_alias_node() {
    int sp = sav();
    if (peek() != '*') return false;
    pos++;
    if (!ns_anchor_name()) { rst(sp); return false; }
    return true;
}

static bool ns_flow_yaml_node(int n, Ctx c) {
    int sp = sav();
    if (c_ns_alias_node()) return true; rst(sp);
    if (ns_flow_yaml_content(n, c)) return true; rst(sp);
    if (c_ns_properties(n, c)) {
        int sp2 = sav();
        if (s_separate(n, c) && ns_flow_yaml_content(n, c)) return true;
        rst(sp2);
        return true;
    }
    rst(sp);
    return false;
}
static bool c_flow_json_node(int n, Ctx c) {
    int sp = sav();
    int sp2 = sav();
    if (c_ns_properties(n, c) && s_separate(n, c)) {
    } else { rst(sp2); }
    if (c_flow_json_content(n, c)) return true;
    rst(sp);
    return false;
}
static bool ns_flow_node(int n, Ctx c) {
    int sp = sav();
    if (c_ns_alias_node()) return true; rst(sp);
    if (ns_flow_content(n, c)) return true; rst(sp);
    if (c_ns_properties(n, c)) {
        int sp2 = sav();
        if (s_separate(n, c) && ns_flow_content(n, c)) return true;
        rst(sp2);
        return true;
    }
    rst(sp);
    return false;
}

static bool ns_s_flow_seq_entries(int n, Ctx c) {
    if (at_forbidden()) return false;
    if (!ns_flow_seq_entry(n, c)) return false;
    int sp = sav();
    if (!s_separate(n, c)) rst(sp);
    if (peek() == ',') {
        pos++;
        int sp2 = sav();
        if (!s_separate(n, c)) rst(sp2);
        int sp3 = sav();
        if (!ns_s_flow_seq_entries(n, c)) rst(sp3);
    }
    return true;
}
static bool c_flow_sequence(int n, Ctx c) {
    int sp = sav();
    if (peek() != '[') return false;
    pos++;
    int sp2 = sav();
    if (!s_separate(n, c)) rst(sp2);
    int sp3 = sav();
    Ctx ic = flow_in_ctx(c);
    if (!ns_s_flow_seq_entries(n, ic)) rst(sp3);
    int sp4 = sav();
    if (!s_separate(n, ic)) rst(sp4);
    if (peek() != ']') { rst(sp); return false; }
    pos++;
    return true;
}
static bool ns_flow_seq_entry(int n, Ctx c) {
    int sp = sav();
    if (ns_flow_pair(n, c)) return true; rst(sp);
    return ns_flow_node(n, c);
}

static bool ns_flow_map_yaml_key_entry(int n, Ctx c) {
    int sp = sav();
    if (!ns_flow_yaml_node(n, c)) return false;
    int sp2 = sav();
    int sp3 = sav();
    if (!s_separate(n, c)) rst(sp3);
    if (c_ns_flow_map_separate_value(n, c)) return true;
    rst(sp2);
    return true;
}
static bool c_ns_flow_map_empty_key_entry(int n, Ctx c) { return c_ns_flow_map_separate_value(n, c); }
static bool c_ns_flow_map_json_key_entry(int n, Ctx c) {
    int sp = sav();
    if (!c_flow_json_node(n, c)) return false;
    int sp2 = sav();
    int sp3 = sav();
    if (!s_separate(n, c)) rst(sp3);
    if (c_ns_flow_map_adjacent_value(n, c)) return true;
    rst(sp2);
    return true;
}
static bool ns_flow_map_implicit_entry(int n, Ctx c) {
    int sp = sav();
    if (ns_flow_map_yaml_key_entry(n, c)) return true; rst(sp);
    if (c_ns_flow_map_json_key_entry(n, c)) return true; rst(sp);
    return c_ns_flow_map_empty_key_entry(n, c);
}
static bool c_ns_flow_map_separate_value(int n, Ctx c) {
    int sp = sav();
    if (peek() != ':') return false;
    pos++;
    int ch = peek();
    bool unsafe = is_nschar(ch);
    if (unsafe && (c == FI || c == FK) && is_flowind(ch)) unsafe = false;
    if (unsafe) { rst(sp); return false; }
    int sp2 = sav();
    if (s_separate(n, c) && ns_flow_node(n, c)) return true;
    rst(sp2);
    return true;
}
static bool c_ns_flow_map_adjacent_value(int n, Ctx c) {
    int sp = sav();
    if (peek() != ':') return false;
    pos++;
    int sp2 = sav();
    int sp3 = sav();
    if (!s_separate(n, c)) rst(sp3);
    if (ns_flow_node(n, c)) return true;
    rst(sp2);
    return true;
}
static bool ns_flow_map_explicit_entry(int n, Ctx c) {
    int sp = sav();
    if (ns_flow_map_implicit_entry(n, c)) return true;
    rst(sp);
    return true;
}
static bool ns_flow_map_entry(int n, Ctx c) {
    int sp = sav();
    if (peek() == '?') {
        int sp1 = sav();
        pos++;
        if (s_separate(n, c)) {
            if (ns_flow_map_explicit_entry(n, c)) return true;
        }
        rst(sp1);
    }
    return ns_flow_map_implicit_entry(n, c);
}
static bool ns_s_flow_map_entries(int n, Ctx c) {
    if (at_forbidden()) return false;
    if (!ns_flow_map_entry(n, c)) return false;
    int sp = sav();
    if (!s_separate(n, c)) rst(sp);
    if (peek() == ',') {
        pos++;
        int sp2 = sav();
        if (!s_separate(n, c)) rst(sp2);
        int sp3 = sav();
        if (!ns_s_flow_map_entries(n, c)) rst(sp3);
    }
    return true;
}
static bool c_flow_mapping(int n, Ctx c) {
    int sp = sav();
    if (peek() != '{') return false;
    pos++;
    int sp2 = sav();
    if (!s_separate(n, c)) rst(sp2);
    int sp3 = sav();
    Ctx ic = flow_in_ctx(c);
    if (!ns_s_flow_map_entries(n, ic)) rst(sp3);
    int sp4 = sav();
    if (!s_separate(n, ic)) rst(sp4);
    if (peek() != '}') { rst(sp); return false; }
    pos++;
    return true;
}

static bool ns_s_implicit_yaml_key(Ctx c) {
    int start = sav();
    if (!ns_flow_yaml_node(0, c)) return false;
    int sp = sav();
    if (!s_separate_in_line()) rst(sp);
    if (sav() - start > 1024) return false;
    return true;
}
static bool c_s_implicit_json_key(Ctx c) {
    int start = sav();
    if (!c_flow_json_node(0, c)) return false;
    int sp = sav();
    if (!s_separate_in_line()) rst(sp);
    if (sav() - start > 1024) return false;
    return true;
}
static bool ns_flow_pair_yaml_key_entry(int n, Ctx c) {
    int sp = sav();
    if (!ns_s_implicit_yaml_key(FK)) return false;
    if (!c_ns_flow_map_separate_value(n, c)) { rst(sp); return false; }
    return true;
}
static bool c_ns_flow_pair_json_key_entry(int n, Ctx c) {
    int sp = sav();
    if (!c_s_implicit_json_key(FK)) return false;
    if (!c_ns_flow_map_adjacent_value(n, c)) { rst(sp); return false; }
    return true;
}
static bool ns_flow_pair_entry(int n, Ctx c) {
    int sp = sav();
    if (ns_flow_pair_yaml_key_entry(n, c)) return true; rst(sp);
    if (c_ns_flow_pair_json_key_entry(n, c)) return true; rst(sp);
    return c_ns_flow_map_empty_key_entry(n, c);
}
static bool ns_flow_pair(int n, Ctx c) {
    int sp = sav();
    if (peek() == '?') {
        int sp1 = sav();
        pos++;
        if (s_separate(n, c) && ns_flow_map_explicit_entry(n, c)) return true;
        rst(sp1);
    }
    return ns_flow_pair_entry(n, c);
}

struct BlockHeader { int m; Chomp t; bool ok; };
static BlockHeader c_b_block_header() {
    BlockHeader h{0, CCLIP, false};
    int sp = sav();
    auto parseIndent = [&]() -> int { int c = peek(); if (c >= '1' && c <= '9') { pos++; return c - '0'; } return 0; };
    auto parseChomp = [&]() -> Chomp {
        int c = peek();
        if (c == '-') { pos++; return CSTRIP; }
        if (c == '+') { pos++; return CKEEP; }
        return CCLIP;
    };
    int sp2 = sav();
    int m = parseIndent();
    Chomp t = parseChomp();
    if (m == 0) {
        rst(sp2);
        t = parseChomp();
        m = parseIndent();
    }
    h.m = m;
    h.t = t;
    if (!s_b_comment()) { rst(sp); return h; }
    h.ok = true;
    return h;
}
static int auto_detect_indent_m(int n) {
    int sp = sav();
    int maxLeading = 0;
    int firstNonEmpty = -1;
    while (true) {
        int sp2 = sav();
        int spaces = 0;
        while (peek() == ' ') { pos++; spaces++; }
        if (b_break() || eof()) {
            if (spaces > maxLeading) maxLeading = spaces;
            if (eof()) break;
            continue;
        }
        firstNonEmpty = spaces;
        break;
    }
    rst(sp);
    if (firstNonEmpty < 0) {
        int m = maxLeading - n;
        if (m < 1) m = 1;
        return m;
    }
    if (maxLeading > firstNonEmpty) return -1;
    int m = firstNonEmpty - n;
    if (m < 1) m = 1;
    return m;
}

static bool b_chomped_last(Chomp t) { if (eof()) return true; return b_break(); }
static bool l_trail_comments(int n) {
    int sp = sav();
    s_indent_lt(n);
    if (!c_nb_comment_text()) { rst(sp); return false; }
    if (!b_comment()) { rst(sp); return false; }
    while (true) { int sp2 = sav(); if (!l_comment()) { rst(sp2); break; } }
    return true;
}
static bool l_strip_empty(int n) {
    while (true) {
        int sp = sav();
        s_indent_le(n);
        if (!b_non_content()) { rst(sp); break; }
    }
    int sp = sav();
    if (!l_trail_comments(n)) rst(sp);
    return true;
}
static bool l_keep_empty(int n) {
    while (true) { int sp = sav(); if (!l_empty(n, BI)) { rst(sp); break; } }
    int sp = sav();
    if (!l_trail_comments(n)) rst(sp);
    return true;
}
static bool l_chomped_empty(int n, Chomp t) { if (t == CKEEP) return l_keep_empty(n); return l_strip_empty(n); }
static bool l_nb_literal_text(int n) {
    while (true) { int sp = sav(); if (!l_empty(n, BI)) { rst(sp); break; } }
    int sp = sav();
    if (at_forbidden()) { rst(sp); return false; }
    if (!s_indent(n)) { rst(sp); return false; }
    if (!is_nbchar(peek())) { rst(sp); return false; }
    while (is_nbchar(peek())) pos++;
    return true;
}
static bool b_nb_literal_next(int n) {
    int sp = sav();
    if (!b_as_line_feed()) return false;
    if (!l_nb_literal_text(n)) { rst(sp); return false; }
    return true;
}
static bool l_literal_content(int n, Chomp t) {
    int sp = sav();
    int sp2 = sav();
    if (l_nb_literal_text(n)) {
        while (true) { int sp3 = sav(); if (!b_nb_literal_next(n)) { rst(sp3); break; } }
        if (!b_chomped_last(t)) { rst(sp); return false; }
    } else { rst(sp2); }
    return l_chomped_empty(n, t);
}
static bool c_l_literal(int n) {
    int sp = sav();
    if (peek() != '|') return false;
    pos++;
    BlockHeader h = c_b_block_header();
    if (!h.ok) { rst(sp); return false; }
    int m = h.m;
    if (m == 0) {
        m = auto_detect_indent_m(n);
        if (m < 0) { rst(sp); return false; }
    }
    if (m < 1) m = 1;
    if (!l_literal_content(n + m, h.t)) { rst(sp); return false; }
    return true;
}

static bool s_nb_folded_text(int n) {
    int sp = sav();
    if (at_forbidden()) { rst(sp); return false; }
    if (!s_indent(n)) return false;
    if (!is_nschar(peek())) { rst(sp); return false; }
    pos++;
    while (is_nbchar(peek())) pos++;
    return true;
}
static bool l_nb_folded_lines(int n) {
    if (!s_nb_folded_text(n)) return false;
    while (true) {
        int sp = sav();
        if (!b_l_folded(n, BI)) { rst(sp); break; }
        if (!s_nb_folded_text(n)) { rst(sp); break; }
    }
    return true;
}
static bool s_nb_spaced_text(int n) {
    int sp = sav();
    if (at_forbidden()) { rst(sp); return false; }
    if (!s_indent(n)) return false;
    if (!is_white(peek())) { rst(sp); return false; }
    pos++;
    while (is_nbchar(peek())) pos++;
    return true;
}
static bool b_l_spaced(int n) {
    if (!b_as_line_feed()) return false;
    while (true) { int sp = sav(); if (!l_empty(n, BI)) { rst(sp); break; } }
    return true;
}
static bool l_nb_spaced_lines(int n) {
    if (!s_nb_spaced_text(n)) return false;
    while (true) {
        int sp = sav();
        if (!b_l_spaced(n)) { rst(sp); break; }
        if (!s_nb_spaced_text(n)) { rst(sp); break; }
    }
    return true;
}
static bool l_nb_same_lines(int n) {
    while (true) { int sp = sav(); if (!l_empty(n, BI)) { rst(sp); break; } }
    int sp = sav();
    if (l_nb_folded_lines(n)) return true;
    rst(sp);
    return l_nb_spaced_lines(n);
}
static bool l_nb_diff_lines(int n) {
    if (!l_nb_same_lines(n)) return false;
    while (true) {
        int sp = sav();
        if (!b_as_line_feed()) { rst(sp); break; }
        if (!l_nb_same_lines(n)) { rst(sp); break; }
    }
    return true;
}
static bool l_folded_content(int n, Chomp t) {
    int sp = sav();
    int sp2 = sav();
    if (l_nb_diff_lines(n)) {
        if (!b_chomped_last(t)) { rst(sp); return false; }
    } else { rst(sp2); }
    return l_chomped_empty(n, t);
}
static bool c_l_folded(int n) {
    int sp = sav();
    if (peek() != '>') return false;
    pos++;
    BlockHeader h = c_b_block_header();
    if (!h.ok) { rst(sp); return false; }
    int m = h.m;
    if (m == 0) {
        m = auto_detect_indent_m(n);
        if (m < 0) { rst(sp); return false; }
    }
    if (m < 1) m = 1;
    if (!l_folded_content(n + m, h.t)) { rst(sp); return false; }
    return true;
}

static bool c_l_block_seq_entry(int n) {
    int sp = sav();
    if (peek() != '-') return false;
    pos++;
    if (is_nschar(peek())) { rst(sp); return false; }
    if (!s_l_block_indented(n, BI)) { rst(sp); return false; }
    return true;
}
static bool ns_l_compact_sequence(int n) {
    if (!c_l_block_seq_entry(n)) return false;
    while (true) {
        int sp = sav();
        if (!s_indent(n)) { rst(sp); break; }
        if (!c_l_block_seq_entry(n)) { rst(sp); break; }
    }
    return true;
}

static bool c_l_block_map_explicit_key(int n);
static bool l_block_map_explicit_value(int n);
static bool c_l_block_map_explicit_entry(int n) {
    int sp = sav();
    if (!c_l_block_map_explicit_key(n)) return false;
    int sp2 = sav();
    if (!l_block_map_explicit_value(n)) rst(sp2);
    return true;
}
static bool c_l_block_map_explicit_key(int n) {
    int sp = sav();
    if (peek() != '?') return false;
    pos++;
    if (is_nschar(peek())) { rst(sp); return false; }
    if (!s_l_block_indented(n, BO)) { rst(sp); return false; }
    return true;
}
static bool l_block_map_explicit_value(int n) {
    int sp = sav();
    if (!s_indent(n)) return false;
    if (peek() != ':') { rst(sp); return false; }
    pos++;
    if (is_nschar(peek())) { rst(sp); return false; }
    if (!s_l_block_indented(n, BO)) { rst(sp); return false; }
    return true;
}
static bool ns_s_block_map_implicit_key() {
    int sp = sav();
    if (c_s_implicit_json_key(BK)) return true; rst(sp);
    return ns_s_implicit_yaml_key(BK);
}
static bool c_l_block_map_implicit_value(int n) {
    int sp = sav();
    if (peek() != ':') return false;
    pos++;
    int sp2 = sav();
    if (s_l_block_node(n, BO)) return true;
    rst(sp2);
    if (s_l_comments()) return true;
    rst(sp); return false;
}
static bool ns_l_block_map_implicit_entry(int n) {
    int sp = sav();
    int sp2 = sav();
    if (!ns_s_block_map_implicit_key()) rst(sp2);
    if (!c_l_block_map_implicit_value(n)) { rst(sp); return false; }
    return true;
}
static bool ns_l_block_map_entry(int n) {
    int sp = sav();
    if (c_l_block_map_explicit_entry(n)) return true; rst(sp);
    return ns_l_block_map_implicit_entry(n);
}
static bool ns_l_compact_mapping(int n) {
    if (!ns_l_block_map_entry(n)) return false;
    while (true) {
        int sp = sav();
        if (!s_indent(n)) { rst(sp); break; }
        if (!ns_l_block_map_entry(n)) { rst(sp); break; }
    }
    return true;
}

static bool l_block_sequence(int n) {
    if (!atSOL()) return false;
    int sp0 = sav();
    int leading = 0;
    while (peek() == ' ') { pos++; leading++; }
    rst(sp0);
    int target = leading;
    if (target < n + 1) return false;
    int matched = 0;
    while (true) {
        int sp = sav();
        if (at_forbidden()) { rst(sp); break; }
        if (!s_indent(target)) { rst(sp); break; }
        if (!c_l_block_seq_entry(target)) { rst(sp); break; }
        matched++;
    }
    return matched >= 1;
}
static bool l_block_mapping(int n) {
    if (!atSOL()) return false;
    int sp0 = sav();
    int leading = 0;
    while (peek() == ' ') { pos++; leading++; }
    rst(sp0);
    int target = leading;
    if (target < n + 1) return false;
    int matched = 0;
    while (true) {
        int sp = sav();
        if (at_forbidden()) { rst(sp); break; }
        if (!s_indent(target)) { rst(sp); break; }
        if (!ns_l_block_map_entry(target)) { rst(sp); break; }
        matched++;
    }
    return matched >= 1;
}

static bool s_l_block_indented(int n, Ctx c) {
    int sp = sav();
    {
        int sp2 = sav();
        int m = 0;
        while (peek() == ' ') { pos++; m++; }
        if (m > 0) {
            int sp3 = sav();
            if (ns_l_compact_sequence(n + 1 + m)) return true;
            rst(sp3);
            if (ns_l_compact_mapping(n + 1 + m)) return true;
        }
        rst(sp2);
    }
    if (s_l_block_node(n, c)) return true;
    rst(sp);
    if (s_l_comments()) return true;
    rst(sp);
    return false;
}

static bool s_l_flow_in_block(int n) {
    int sp = sav();
    if (!s_separate(n + 1, FO)) return false;
    if (!ns_flow_node(n + 1, FO)) { rst(sp); return false; }
    if (!s_l_comments()) { rst(sp); return false; }
    return true;
}
static bool s_l_block_scalar(int n, Ctx c) {
    int sp = sav();
    if (!s_separate(n + 1, c)) return false;
    int sp2 = sav();
    if (c_ns_properties(n + 1, c)) {
        if (!s_separate(n + 1, c)) { rst(sp2); }
    } else { rst(sp2); }
    int sp3 = sav();
    if (c_l_literal(n)) return true; rst(sp3);
    if (c_l_folded(n)) return true;
    rst(sp); return false;
}
static bool seq_space(int n, Ctx c) {
    if (c == BO) return l_block_sequence(n - 1);
    return l_block_sequence(n);
}
// With backtracking: try with-props, then without-props
static bool s_l_block_collection(int n, Ctx c) {
    int sp = sav();
    // Try with optional (s-separate + props)
    {
        int sp2 = sav();
        if (s_separate(n + 1, c) && c_ns_properties(n + 1, c)) {
            int sp3 = sav();
            if (s_l_comments()) {
                int sp4 = sav();
                if (seq_space(n, c)) return true;
                rst(sp4);
                if (l_block_mapping(n)) return true;
            }
            rst(sp3);
        }
        rst(sp2);
    }
    // Try without props
    if (!s_l_comments()) { rst(sp); return false; }
    int sp4 = sav();
    if (seq_space(n, c)) return true;
    rst(sp4);
    if (l_block_mapping(n)) return true;
    rst(sp); return false;
}
static bool s_l_block_in_block(int n, Ctx c) {
    int sp = sav();
    if (s_l_block_scalar(n, c)) return true; rst(sp);
    return s_l_block_collection(n, c);
}
static bool s_l_block_node(int n, Ctx c) {
    int sp = sav();
    if (s_l_block_in_block(n, c)) return true; rst(sp);
    return s_l_flow_in_block(n);
}

static bool c_byte_order_mark() { if (peek() == 0xFEFF) { pos++; return true; } return false; }
static bool l_document_prefix() {
    int sp = sav();
    if (!c_byte_order_mark()) rst(sp);
    while (true) { int sp2 = sav(); if (!l_comment()) { rst(sp2); break; } }
    return true;
}
static bool c_directives_end() {
    if (peek() == '-' && peek(1) == '-' && peek(2) == '-') { pos += 3; return true; }
    return false;
}
static bool c_document_end() {
    if (peek() == '.' && peek(1) == '.' && peek(2) == '.') {
        int next = peek(3);
        if (is_nschar(next)) return false;
        pos += 3;
        return true;
    }
    return false;
}
static bool l_document_suffix() {
    int sp = sav();
    if (!c_document_end()) return false;
    if (!s_l_comments()) { rst(sp); return false; }
    return true;
}
static bool l_bare_document() { return s_l_block_node(-1, BI); }
static bool l_explicit_document() {
    int sp = sav();
    if (!c_directives_end()) return false;
    int sp2 = sav();
    if (l_bare_document()) return true;
    rst(sp2);
    if (s_l_comments()) return true;
    rst(sp); return false;
}
static bool l_directive_document() {
    int sp = sav();
    if (!l_directive()) return false;
    while (true) { int sp2 = sav(); if (!l_directive()) { rst(sp2); break; } }
    if (!l_explicit_document()) { rst(sp); return false; }
    return true;
}
static bool l_any_document() {
    int sp = sav();
    if (l_directive_document()) return true; rst(sp);
    if (l_explicit_document()) return true; rst(sp);
    return l_bare_document();
}

static bool l_yaml_stream() {
    while (true) {
        int sp = sav();
        if (!l_document_prefix()) { rst(sp); break; }
        if (sp == sav()) break;
    }
    reset_doc_state();
    int sp0 = sav();
    if (!l_any_document()) rst(sp0);
    while (true) {
        int sp2 = sav();
        if (l_document_suffix()) {
            while (true) { int sp3 = sav(); if (!l_document_suffix()) { rst(sp3); break; } }
            while (true) { int sp3 = sav(); if (!l_document_prefix()) { rst(sp3); break; } if (sp3 == sav()) break; }
            reset_doc_state();
            int sp4 = sav();
            if (!l_any_document()) rst(sp4);
            continue;
        }
        rst(sp2);
        if (c_byte_order_mark()) continue;
        rst(sp2);
        if (l_comment()) continue;
        rst(sp2);
        reset_doc_state();
        if (l_explicit_document()) continue;
        rst(sp2);
        break;
    }
    return eof();
}

int main() {
    string in((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());
    int i = 0;
    bool ok_utf = true;
    if (in.size() >= 3 && (uint8_t)in[0] == 0xEF && (uint8_t)in[1] == 0xBB && (uint8_t)in[2] == 0xBF) {
        i = 3;
        S.push_back(0xFEFF);
    }
    while (i < (int)in.size()) {
        uint8_t b = in[i];
        int cp; int n;
        if (b < 0x80) { cp = b; n = 1; }
        else if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; n = 2; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; n = 3; }
        else if ((b & 0xF8) == 0xF0) { cp = b & 0x07; n = 4; }
        else { ok_utf = false; break; }
        if (i + n > (int)in.size()) { ok_utf = false; break; }
        bool bad = false;
        for (int j = 1; j < n; j++) {
            uint8_t cc = in[i+j];
            if ((cc & 0xC0) != 0x80) { bad = true; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (bad) { ok_utf = false; break; }
        S.push_back(cp);
        i += n;
    }
    N = (int)S.size();
    pos = 0;
    bool ok = ok_utf && l_yaml_stream();
    cout << (ok ? "valid" : "invalid");
    return 0;
}
