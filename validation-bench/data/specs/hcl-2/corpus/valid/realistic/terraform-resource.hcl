resource "aws_instance" "web" {
  ami           = "ami-0123456789abcdef0"
  instance_type = var.instance_type
  count         = length(var.subnets)
  tags = {
    Name        = "web-${count.index}"
    Environment = local.env
  }
}
