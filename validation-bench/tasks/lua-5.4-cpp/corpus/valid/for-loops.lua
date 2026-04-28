for i = 1, 10 do
  print(i)
end

for i = 10, 1, -1 do
  print(i)
end

local t = {1, 2, 3}
for k, v in ipairs(t) do
  print(k, v)
end

for k, v in pairs({a = 1, b = 2}) do
  print(k, v)
end
