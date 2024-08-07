// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "simulation-tools",
    platforms: [
        .iOS(.v14),
        .macOS(.v11),
        .macCatalyst(.v14)
    ],
    products: [
        .library(
            name: "SimulationTools",
            targets: ["SimulationTools"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/eugenebokhan/metal-tools",
            .upToNextMajor(from: "1.2.0")
        )
    ],
    targets: [
        .target(
            name: "SimulationToolsSharedTypes",
            publicHeadersPath: "."
        ),
        .target(
            name: "SimulationTools",
            dependencies: [
                .product(name: "MetalTools", package: "metal-tools"),
                .target(name: "SimulationToolsSharedTypes")
            ],
            resources: [
                .process("CollisionDetection/BroadPhase/BitonicSort/BitonicSort.metal"),
                .process("CollisionDetection/BroadPhase/SpatialHashing.metal"),
            ]
        ),
        .testTarget(
            name: "SimulationToolsTests",
            dependencies: ["SimulationTools"]
        ),
    ]
)
