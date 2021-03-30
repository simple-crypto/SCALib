# Container for building the portable wheel
FROM quay.io/pypa/manylinux2014_x86_64
RUN yum install -y boost-devel
WORKDIR /root
RUN curl -O https://static.rust-lang.org/dist/rust-1.50.0-x86_64-unknown-linux-gnu.tar.gz \
	&& tar xzf rust-1.50.0-x86_64-unknown-linux-gnu.tar.gz \
	&& rust-1.50.0-x86_64-unknown-linux-gnu/install.sh --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu \
	&& rm -rf rust-1.50.0-x86_64-unknown-linux-gnu \
	&& rm -rf rust-1.50.0-x86_64-unknown-linux-gnu.tar.gz

RUN mkdir /cargo_home && chmod 777 /cargo_home
ENV CARGO_HOME=/cargo_home

ENV PATH /opt/python/cp36-cp36m/bin/:/opt/python/cp37-cp37m/bin/:/opt/python/cp38-cp38/bin/:/opt/python/cp39-cp39/bin/:$PATH
RUN pip3.6 install --no-cache-dir -U setuptools setuptools-rust wheel

WORKDIR /io

ENTRYPOINT []
CMD export PATH="$/root/.cargo/bin:$PATH" \
		&& export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.cargo/lib" \
		&& python3.6 setup.py bdist_wheel --py-limited-api=cp35 \
		&& for whl in dist/*-cp35-abi3-linux_x86_64.whl; do auditwheel repair -w dist --plat manylinux2014_x86_64 "$whl"; done

