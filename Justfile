default:
    podman build -t tem-segmentation-etl .
    podman run --rm -v "$(pwd):/work" -w /work tem-segmentation-etl make pipeline

pipeline:
    podman build -t tem-segmentation-etl .
    podman run --rm -v "$(pwd):/work" -w /work tem-segmentation-etl make pipeline

check:
    podman run --rm -v "$(pwd):/work" -w /work tem-segmentation-etl make check