
all: gpu_image

clean:
	rm -f gpu_image
	rm LennaAnim*

CFLAGS = --ptxas-options=-v

debug: CFLAGS += -g -G -Xcompiler -rdynamic -lineinfo
debug: gpu_image

gpu_image: image.cu
	nvcc $(CFLAGS) image.cu -o gpu_image

mod7: gpu_image
	convert goomba-small.png goomba-small.ppm
	./gpu_image -m 6 -p -i Lenna.ppm -o LennaAnim -k 0 -M goomba-small.ppm

# Debug:	mogrify -format png LennaAnim*.ppm
	convert -delay 30 -loop 0 LennaAnim*.ppm LennaAnim.gif
mod6: gpu_image
	convert Lenna.png Lenna.ppm
	@echo "Encode 'Hello Hidden World!'"
	./gpu_image -i Lenna.ppm -o Lenna.hello.ppm -k 0 -m 4 -M "Hello Hidden World!"
	convert Lenna.hello.ppm Lenna.hello.png
	convert Lenna.hello.png Lenna.hello.conv.ppm
	@echo "Reading back message:"
	./gpu_image -i Lenna.ppm -o Lenna.hello.conv.ppm -k 0 -m 5

	@echo "\n\nEncode 'All your base are belong to us'"
	./gpu_image -i Lenna.ppm -o Lenna.base.ppm -k 4 -m 4 -M "All your base are belong to us"
	convert Lenna.base.ppm Lenna.base.png
	convert Lenna.base.png Lenna.base.conv.ppm
	@echo "Reading back message:"
	./gpu_image -i Lenna.ppm -o Lenna.base.conv.ppm -k 4 -m 5





mod5: gpu_image
	convert Lenna.png Lenna.ppm
	@echo "Constant Memory, Apply or-mask of 0xFF (set red channel to max)"
	./gpu_image -i Lenna.ppm -o Lenna.parsed2.ppm -k 255
	convert Lenna.parsed2.ppm Lenna.parsed2.png

	@echo "Constant Memory, Apply and-mask of 0xFF (show red component only)"
	./gpu_image -i Lenna.ppm -o Lenna.parsed3.ppm -k 255 -m 1
	convert Lenna.parsed3.ppm Lenna.parsed3.png

	@echo "Flip Image Horizontally (work done using shared memory)"
	./gpu_image -i Lenna.ppm -o Lenna.horflip.ppm -m 2
	convert Lenna.horflip.ppm Lenna.horflip.png

	@echo "Flip Image Vertically (work done using shared memory)"
	./gpu_image -i Lenna.ppm -o Lenna.verflip.ppm -m 3
	convert Lenna.verflip.ppm Lenna.verflip.png


mod4: gpu_image
	convert Lenna.png Lenna.ppm
	convert colorbands.png colorbands.ppm
	convert pluto.jpg pluto.ppm
	convert sun.jpg sun.ppm

	@echo "Pageable Memory Test (512x512):\n"
	time ./gpu_image -i Lenna.ppm -o Lenna.parsed.ppm -n 10
	@echo "\nPinned Memory Test (512x512):\n"
	time ./gpu_image -i Lenna.ppm -o Lenna.parsed.ppm -n 10 -p
	@echo "Pageable Memory Test (1024x1024):\n"
	time ./gpu_image -i colorbands.ppm -o colorbands.parsed.ppm -n 10
	@echo "\nPinned Memory Test (1024x1024):\n"
	time ./gpu_image -i colorbands.ppm -o colorbands.parsed.ppm -n 10 -p

	@echo "Pageable Memory Test (2048):\n"
	time ./gpu_image -i pluto.ppm -o pluto.parsed.ppm -n 10 -t 1024
	@echo "\nPinned Memory Test (2048):\n"
	time ./gpu_image -i pluto.ppm -o pluto.parsed.ppm -n 10 -p -t 1024
	@echo "Pageable Memory Test (4096):\n"
	time ./gpu_image -i sun.ppm -o sun.parsed.ppm -n 10 -t 1024
	@echo "\nPinned Memory Test (4096):\n"
	time ./gpu_image -i sun.ppm -o sun.parsed.ppm -n 10 -p -t 1024

	convert Lenna.parsed.ppm Lenna.parsed.png
	convert colorbands.parsed.ppm colorbands.parsed.png
	convert pluto.parsed.ppm pluto.parsed.png
	convert sun.parsed.ppm sun.parsed.png

