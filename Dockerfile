FROM nvidia/cudagl:11.1.1-devel-ubuntu18.04

# Configuration
ENV PROJDIR=/gravis
ENV TZ=Europe/Berlin
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
	apt-get update && \
	apt-get install --no-install-recommends -y \
	apt-utils \
	cmake \
	git \
	xvfb \
	wget

RUN git clone --recursive https://github.com/aida-ugent/graph-vis-eval.git $PROJDIR

# FR-RTX
RUN apt-get install --no-install-recommends -y \
	freeglut3-dev \
	gcc-6 \
	g++-6 \
	libboost-all-dev \
	libglew-dev \
	libglfw3 \
	libglfw3-dev \
	libtbb-dev \
	libpng-dev \
	libtiff-dev \
	libxcursor-dev \
	libxinerama-dev \
	libxrandr-dev 

RUN update-alternatives \
	--install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
	--slave /usr/bin/g++ g++ /usr/bin/g++-6

# Download nvidia optix 7.1.0 from https://developer.nvidia.com/designworks/optix/downloads/legacy
ADD nvidia-optix-7-1-0 nvidia-optix-7-1-0
WORKDIR /nvidia-optix-7-1-0/SDK
RUN mkdir build && \
	cd build && \
	cmake -j 16 .. && \
	make -j8

WORKDIR $PROJDIR/methods/frrtx
ENV CUDA_PATH=/usr/local/cuda-11.1
RUN mkdir build && \
	cd build && \
	cmake -j 16 .. \
	-DCMAKE_BUILD_TYPE=Release \
	-DOptiX_INCLUDE:PATH=/nvidia-optix-7-1-0/include \
	-DBIN2C=/usr/local/cuda-11.1/bin/bin2c  \
	-DCMAKE_LIBRARY_PATH=/usr/local/cuda-11.1/lib64/stubs && \
	make -j8

# GLAM
RUN apt-get install --no-install-recommends -y \
	nvidia-opencl-dev \
	libcgal-dev

WORKDIR $PROJDIR/tools/glam
RUN mkdir build && \
	cd build && \
	cmake -j 16 .. \
	-DCMAKE_BUILD_TYPE=Release \
	-DOpenCL_INCLUDE_DIR:PATH=/usr/local/lib/intel-opencl && \
	make -j8

# DRGraph
RUN apt-get install --no-install-recommends -y \
	autotools-dev \
	build-essential \
	g++ \
	python-dev \
	libicu-dev \
	libbz2-dev \
	libgsl-dev=2.4+dfsg-6

WORKDIR /
ENV BOOST_VERSION=1_58_0
RUN wget https://sourceforge.net/projects/boost/files/boost/1.58.0/boost_$BOOST_VERSION.tar.gz && \
	tar -xvf /boost_$BOOST_VERSION.tar.gz && \
	rm boost_$BOOST_VERSION.tar.gz

WORKDIR /boost_$BOOST_VERSION
RUN ./bootstrap.sh && \
	./b2 --with=all -j 16 --prefix=/usr/local install	 

WORKDIR $PROJDIR/methods/drgraph
RUN mkdir build && \
	cd build && \
	unset DRGRAPH_GPU_COMPILE && \
	cmake -j 16 .. -DBoost_INCLUDE_DIR=/usr/local/include && \
	make -j8

# Python environments
RUN apt-get install --no-install-recommends -y \
	python-setuptools \
	python3-pip \
	python3-tk \
	python-pip \
	python-dev && \
	pip install virtualenv && \
	python3 -m pip install --user virtualenv

WORKDIR $PROJDIR/methods
RUN virtualenv -p /usr/bin/python arope_venv && \
	arope_venv/bin/pip install -r arope_requirements.txt && \
	arope_venv/bin/python arope_setup.py install

ENV LD_LIBRARY_PATH="" 
WORKDIR $PROJDIR/methods/deepwalk
RUN virtualenv -p /usr/bin/python ../deepwalk_venv && \
	../deepwalk_venv/bin/pip install -r ../deepwalk_requirements.txt && \
	../deepwalk_venv/bin/python setup.py install

WORKDIR $PROJDIR/methods/gae
RUN virtualenv -p /usr/bin/python ../gae_venv && \
	../gae_venv/bin/pip install -r ../gae_requirements.txt && \
	../gae_venv/bin/python setup.py install

WORKDIR $PROJDIR/tools/evalne
RUN virtualenv -p /usr/bin/python3 ../../gravis_venv && \
	../../gravis_venv/bin/pip3 install -r ../../requirements.txt && \
	../../gravis_venv/bin/python3 setup.py install

RUN rm -rf /var/lib/apt/lists/*

WORKDIR $PROJDIR
RUN echo "source gravis_venv/bin/activate" > ~/.bashrc
ENV DISPLAY=:0.0
CMD nohup Xvfb :0 -screen 0 1024x768x16 >nohup.out 2>&1 & bash