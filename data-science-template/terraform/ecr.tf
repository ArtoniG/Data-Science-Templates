# terraform/ecr.tf

resource "aws_kms_key" "ecr_key" {
  description             = "KMS Key for Credit Decision ECR Encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_ecr_repository" "serving_repo" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "IMMUTABLE" # Prevents overwriting release tags in production

  image_scanning_configuration {
    scan_on_push = true # Automatically scans images for CVEs upon CI/CD push
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr_key.arn
  }

  tags = {
    Name = var.ecr_repository_name
  }
}

# Retain only the last 20 tagged production images to optimize storage costs
resource "aws_ecr_lifecycle_policy" "repo_policy" {
  repository = aws_ecr_repository.serving_repo.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 20 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 20
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}