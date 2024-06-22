// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "simulation-tools",
    platforms: [
        .iOS(.v14),
        .macOS(.v11),
        .macCatalyst(.v13)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
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
            name: "SimulationTools",
            dependencies: [
                .product(name: "MetalComputeTools", package: "metal-tools")
            ],
            resources: [
                .process("SimulationTools/CollisionDetection/BroadPhase/BitonicSort/BitonicSort.metal"),
                .process("SimulationTools/CollisionDetection/BroadPhase/SpatialHashing.metal"),
            ]
        ),
        .testTarget(
            name: "simulation-toolsTests",
            dependencies: ["simulation-tools"]
        ),
    ]
)
