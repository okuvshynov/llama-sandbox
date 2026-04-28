local n = 10
if n > 0 then
  print("positive")
elseif n < 0 then
  print("negative")
else
  print("zero")
end

while n > 0 do
  n = n - 1
end

repeat
  n = n + 1
until n >= 5
