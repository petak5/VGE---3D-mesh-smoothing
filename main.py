import trimesh
import numpy as np
import argparse


def main():
    args = parse_args()

    mesh = trimesh.load(args.input)

    # noisy_mesh = add_noise(mesh, std_dev=0.0001)
    # noisy_mesh.export("noisy.ply")
    # if args.show:
    #     noisy_mesh.show()
    # exit()

    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    adjacency = calculate_adjacency(mesh)

    if args.method == "laplacian":
        smoothed_mesh = laplacian_smoothing(mesh, adjacency, args.lambd, args.iterations)
    elif args.method == "taubin":
        smoothed_mesh = taubin_smoothing(mesh, adjacency, args.lambd, args.mu, args.iterations)
    elif args.method == "bilateral_denoise":
        smoothed_mesh = bilateral_denoise(mesh, adjacency, args.sigma_s, args.sigma_n, args.iterations)
    else:
        raise NotImplementedError(f"Method '{args.method}' not supported")

    print("Exporting mesh...")
    smoothed_mesh.export(args.output)
    print("Done")

    if args.show:
        smoothed_mesh.show()


def parse_args():
    parser = argparse.ArgumentParser(description="3D Mesh Smoothing Tool")

    parser.add_argument('--input', '-i', required=True, help='Input mesh file')
    parser.add_argument('--output', '-o', required=True, help='Output mesh file')

    parser.add_argument('--method', '-m',
                        choices=['laplacian', 'taubin', 'bilateral_denoise'],
                        default='laplacian',
                        help='Smoothing method to use')

    parser.add_argument('--show', '-s', action='store_true', help='Show mesh when processing finishes')

    parser.add_argument('--iterations', '-n', type=int, default=10, help='Number of smoothing iterations')

    # Laplacian and Taubin
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambd', help='Smoothing factor (for Laplacian and Taubin)')

    # Taubin-specific
    parser.add_argument('--mu', type=float, default=-0.53, help='Negative pass factor (for Taubin only)')

    # Bilateral denoise-specific
    parser.add_argument('--sigma_s', type=float, default=0.02, help='Negative pass factor (for Bilateral denoise only)')
    parser.add_argument('--sigma_n', type=float, default=0.2, help='Negative pass factor (for Bilateral denoise only)')

    return parser.parse_args()


def add_noise(mesh, std_dev=0.01, seed=None):
    print("Add noise method")

    noisy_mesh = mesh.copy()

    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(scale=std_dev, size=mesh.vertices.shape)

    noisy_mesh.vertices += noise

    return noisy_mesh


def calculate_adjacency(mesh):
    G = mesh.vertex_adjacency_graph
    adjacency = [[] for _ in range(len(mesh.vertices))]
    for u, v in G.edges():
        adjacency[u].append(v)
        adjacency[v].append(u)

    return adjacency


def laplacian_smoothing(mesh, adjacency, lambd = 0.5, iterations = 5):
    print("Laplacian smoothing method")

    smoothed_mesh = mesh.copy()

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1} of {iterations}")

        vertices = smoothed_mesh.vertices.copy()

        for i, neighbors in enumerate(adjacency):
            if neighbors:
                avg = np.mean(vertices[neighbors], axis=0)
                smoothed_mesh.vertices[i] += lambd * (avg - vertices[i])

    return smoothed_mesh


def taubin_smoothing(mesh, adjacency, lambd=0.5, mu=-0.53, iterations=5):
    print("Taubin smoothing method")

    smoothed_mesh = mesh.copy()

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1} of {iterations}")

        vertices = smoothed_mesh.vertices.copy()

        # First smoothing step (lambda)
        for i, neighbors in enumerate(adjacency):
            if neighbors:
                avg = np.mean(vertices[neighbors], axis=0)
                smoothed_mesh.vertices[i] += lambd * (avg - vertices[i])

        vertices = smoothed_mesh.vertices.copy()

        # Second step (mu, negative offset)
        for i, neighbors in enumerate(adjacency):
            if neighbors:
                avg = np.mean(vertices[neighbors], axis=0)
                smoothed_mesh.vertices[i] += mu * (avg - vertices[i])

    return smoothed_mesh


def bilateral_denoise(mesh, adjacency, sigma_s=0.02, sigma_n=0.2, iterations=1):
    print("Bilateral denoise method")

    denoised_mesh = mesh.copy()

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1} of {iterations}")

        new_denoised_mesh = _bilateral_denoise(denoised_mesh, adjacency, sigma_s, sigma_n)
        denoised_mesh = new_denoised_mesh

    return denoised_mesh


def _bilateral_denoise(mesh, adjacency, sigma_s=0.02, sigma_n=0.2):
    denoised_mesh = mesh.copy()

    sigma_s2 = sigma_s ** 2
    sigma_n2 = sigma_n ** 2


    for i, neighbors in enumerate(adjacency):
        if not neighbors:
            continue

        v_i = mesh.vertices[i]
        n_i = mesh.vertex_normals[i]

        V = mesh.vertices[neighbors]
        N = mesh.vertex_normals[neighbors]

        dv = V - v_i
        dn = N - n_i

        dist2 = np.sum(dv ** 2, axis=1)
        norm_diff2 = np.sum(dn ** 2, axis=1)

        weights = np.exp(-dist2 / sigma_s2) * np.exp(-norm_diff2 / sigma_n2)
        W = weights[:, np.newaxis]

        offset = np.sum(W * dv, axis=0)
        total_weight = np.sum(weights)

        if total_weight > 0:
            denoised_mesh.vertices[i] = v_i + offset / total_weight

    return denoised_mesh


if __name__ == "__main__":
    main()
