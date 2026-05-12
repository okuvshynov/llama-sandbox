source "amazon-ebs" "linux" {
  region        = "us-east-1"
  source_ami    = data.amazon-ami.linux.id
  instance_type = "t3.micro"
  ssh_username  = "ec2-user"
  ami_name      = "myami-${formatdate("YYYY-MM-DD", timestamp())}"
}

build {
  sources = ["source.amazon-ebs.linux"]
}
