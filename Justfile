default:
	podman build -t tem-segmentation-etl .
	podman run --rm -v "$(pwd):/work" -w /work tem-segmentation-etl make all