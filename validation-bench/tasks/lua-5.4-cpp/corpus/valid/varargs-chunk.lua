local function variadic(...)
  local n = select("#", ...)
  return n, ...
end

print(variadic(1, 2, 3))
print(...)
