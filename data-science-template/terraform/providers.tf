# terraform/providers.tf

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Production state should be stored in a remote, encrypted S3 backend
  # backend "s3" {
  #   bucket         = "credit-bureau-tfstate-sa-east-1"
  #   key            = "infrastructure/terraform.tfstate"
  #   region         = "sa-east-1"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Platform    = "Credit-Bureau-Serving-Layer"
      ManagedBy   = "Terraform"
      Compliance  = "LGPD-BACEN"
    }
  }
}