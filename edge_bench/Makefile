bench_opi5p:
	./scripts/bench_linux.sh --backend CPU --threads 4 --repeat 1000 \
		--cmake-args "-DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=ON -DMNN_SEP_BUILD=ON" \
		--output-json output/OPI5P-CPU.json \
		--overwrite --force
	./scripts/bench_linux.sh --backend OPENCL --threads 4 --repeat 1000 \
		--cmake-args "-DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=ON -DMNN_SEP_BUILD=ON" \
		--output-json output/OPI5P-OPENCL.json \
		--overwrite --force

bench_k30p:
	./scripts/bench_android.sh --backend CPU --threads 4 --repeat 1000 \
		--serial adb-e8dca193-RGOds6._adb-tls-connect._tcp \
		--ndk ~/Android/Sdk/ndk/28.0.13004108 \
		--output-json output/K30P-CPU.json \
		--overwrite --force
	./scripts/bench_android.sh --backend OPENCL --threads 4 --repeat 1000 \
		--serial adb-e8dca193-RGOds6._adb-tls-connect._tcp \
		--ndk ~/Android/Sdk/ndk/28.0.13004108 \
		--output-json output/K30P-OPENCL.json \
		--overwrite --force

bench_5800x:
	./scripts/bench_linux.sh --backend CPU --threads 4 --repeat 1000 \
		--output-json output/5800X-CPU.json \
		--overwrite --force
	./scripts/bench_linux.sh --backend CUDA --threads 4 --repeat 1000 \
		--output-json output/3080Ti-CUDA.json \
		--overwrite --force

bench_m4:
	./scripts/bench_macos.sh --backend CPU --threads 4 --repeat 1000 \
		--output-json output/M4-CPU.json \
		--overwrite --force
	./scripts/bench_macos.sh --backend METAL --threads 4 --repeat 1000 \
		--output-json output/M4-METAL.json \
		--overwrite --force

push_5800x:
	rsync -avP data scripts tools CMakeLists.txt Makefile rainy@asus-old:~/dev/OpenSeed-LZU/edge_bench/

push_opi5p:
	rsync -avP data scripts tools CMakeLists.txt Makefile rainy@opi5plus:~/dev/OpenSeed-LZU/edge_bench/

push_m4:
	rsync -avP data scripts tools CMakeLists.txt Makefile rainy@mac:~/dev/OpenSeed-LZU/edge_bench/

pull_m4:
	rsync -avP rainy@mac:~/dev/OpenSeed-LZU/edge_bench/output/M4-CPU.json output/
	rsync -avP rainy@mac:~/dev/OpenSeed-LZU/edge_bench/output/M4-METAL.json output/

pull_opi5p:
	rsync -avP rainy@opi5plus:~/dev/OpenSeed-LZU/edge_bench/output/OPI5P-CPU.json output/
	rsync -avP rainy@opi5plus:~/dev/OpenSeed-LZU/edge_bench/output/OPI5P-OPENCL.json output/

pull_5800x:
	rsync -avP rainy@asus-old:~/dev/OpenSeed-LZU/edge_bench/output/5800X-CPU.json output/
	rsync -avP rainy@asus-old:~/dev/OpenSeed-LZU/edge_bench/output/3080Ti-CUDA.json output/
