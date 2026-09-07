# Source-reconstruction build: the official day-0 image supplies the pinned ABI.
FROM docker.io/lmsysorg/sglang:qwen38flashnext@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae
WORKDIR /sgl-workspace/sglang
COPY patches /opt/qwen-package/patches
COPY upstream-preimages /opt/qwen-package/upstream-preimages
COPY provenance /opt/qwen-package/provenance
COPY scripts/verify_source.py /opt/qwen-package/scripts/verify_source.py
RUN python3 /opt/qwen-package/scripts/verify_source.py --tree /sgl-workspace/sglang --apply --complete
COPY LICENSE NOTICE /usr/share/doc/qwen-tp2-vision/
COPY licenses /usr/share/doc/qwen-tp2-vision/licenses/
RUN python3 -m compileall -q /sgl-workspace/sglang/python/sglang
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
CMD ["--help"]
