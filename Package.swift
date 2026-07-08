// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlxfast-swift", targets: ["MLXFastCLI"]),
        .library(name: "MLXFastCore", targets: ["MLXFastCore"]),
        .library(name: "MLXFastTransform", targets: ["MLXFastTransform"]),
        .library(name: "MLXFastModel", targets: ["MLXFastModel"]),
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
    ],
    dependencies: [
        // Layr-Labs forks: mlx-swift-lm carries the optimized Gemma 4 text tower
        // this benchmark's reference is built on, and it pins mlx-swift to its
        // own fork at branch main -- we must match that requirement (branch main)
        // so SwiftPM resolves a single mlx-swift. Package.resolved pins the exact
        // commit for reproducibility.
        .package(url: "https://github.com/Layr-Labs/mlx-swift", branch: "main"),
        .package(url: "https://github.com/Layr-Labs/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "MLXFastCore"),
        .target(
            name: "MLXFastTransform",
            dependencies: ["MLXFastCore"]
        ),
        .target(
            name: "MLXFastModel",
            dependencies: [
                "MLXFastCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MLXFastCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                "MLXFastHarness",
            ]
        ),
    ]
)
