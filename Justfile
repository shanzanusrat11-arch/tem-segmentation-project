default:
    podman build -t tem-segmentation-emps .
    podman run --rm -e TEM_EPOCHS -v "$(pwd):/work" -w /work tem-segmentation-emps make submission

pipeline:
    podman build -t tem-segmentation-emps .
    podman run --rm -e TEM_EPOCHS -v "$(pwd):/work" -w /work tem-segmentation-emps make submission

check:
    podman run --rm -v "$(pwd):/work" -w /work tem-segmentation-emps make check
