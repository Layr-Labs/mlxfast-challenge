// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlxfast-swift", targets: ["MLXFastCLI"]),
        .executable(
            name: "mlxfast-runtime-worker",
            targets: ["MLXFastRuntimeWorkerCLI"]
        ),
        .library(name: "MLXFastCore", targets: ["MLXFastCore"]),
        .library(name: "MLXFastTransform", targets: ["MLXFastTransform"]),
        .library(name: "MLXFastModel", targets: ["MLXFastModel"]),
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
    ],
    dependencies: [
        // Exact vendored revisions:
        // mlx-swift df1fdc5f7821a1fabe921fdefbc42ac74dcfb6bc
        // mlx-swift-lm bc1c0ee67d15798343be17c9f8f61f7c0d977149
        .package(path: "Vendor/mlx-swift"),
        .package(path: "Vendor/mlx-swift-lm"),
        // TODO(security): Assert the resolved dependency graph before either
        // trusted-harness or participant-worker builds begin.
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
            name: "MLXFastRuntimeWorkerSupport",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXFastHarness"
        ),
        // TODO(security): Pin the trusted-harness source scope independently of
        // participant-controlled package and source manifests.
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXFastTrustedHarness",
            swiftSettings: [
                .define("MLXFAST_TRUSTED_HARNESS")
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
        .executableTarget(
            name: "MLXFastRuntimeWorkerCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastModel",
                "MLXFastRuntimeWorkerSupport",
            ]
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                "MLXFastHarness",
                "MLXFastRuntimeWorkerSupport",
            ]
        ),
    ]
)
