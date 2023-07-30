# FSL code taken from:
# https://github.com/ReproNim/neurodocker/actions/runs/5559312789/jobs/10155292255
# AFNI code from neurodocker generation, but without python-is-python3 package and with new libxp6 link

# docker build --no-cache --tag $USERNAME/img2fmri:$VERSION_NUM --file fsl-afni.Dockerfile .
# ran it from desktop but I believe this also would work:
# docker run -it -p 8888:8888 $USERNAME/img2fmri:$VERSION_NUM

FROM ubuntu:18.04
ENV FSLDIR="/opt/fsl-6.0.6" \
    PATH="/opt/fsl-6.0.6/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl-6.0.6/bin/fsltclsh" \
    FSLWISH="/opt/fsl-6.0.6/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           ca-certificates \
           curl \
           dc \
           file \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libgl1-mesa-dri \
           libglu1-mesa-dev \
           libgomp1 \
           libice6 \
           libopenblas-base \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           nano \
           python3 \
           sudo \
           wget \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Installing FSL ..." \
    && curl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py | python3 - -d /opt/fsl-6.0.6 -V 6.0.6
ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_PLUGINPATH="/opt/afni-latest"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           ca-certificates \
           cmake \
           curl \
           ed \
           gsl-bin \
           libcurl4-openssl-dev \
           libgl1-mesa-dri \
           libglib2.0-0 \
           libglu1-mesa-dev \
           libglw1-mesa \
           libgomp1 \
           libjpeg-turbo8-dev \
           libjpeg62 \
           libssl-dev \
           libudunits2-dev \
           libxm4 \
           multiarch-support \
           netpbm \
           python3-pip \
           tcsh \
           xfonts-base \
           xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && _reproenv_tmppath="$(mktemp -t tmp.XXXXXXXXXX.deb)" \
    && curl -fsSL --retry 5 -o "${_reproenv_tmppath}" https://apt.ligo-wa.caltech.edu:8443/debian/pool/stretch/libxp6/libxp6_1.0.2-2_amd64.deb \
    && apt-get install --yes -q "${_reproenv_tmppath}" \
    && rm "${_reproenv_tmppath}" \
    && _reproenv_tmppath="$(mktemp -t tmp.XXXXXXXXXX.deb)" \
    && curl -fsSL --retry 5 -o "${_reproenv_tmppath}" http://snapshot.debian.org/archive/debian-security/20160113T213056Z/pool/updates/main/libp/libpng/libpng12-0_1.2.49-1%2Bdeb7u2_amd64.deb \
    && apt-get install --yes -q "${_reproenv_tmppath}" \
    && rm "${_reproenv_tmppath}" \
    && apt-get update -qq \
    && apt-get install --yes --quiet --fix-missing \
    && rm -rf /var/lib/apt/lists/* \
    && gsl_path="$(find / -name 'libgsl.so.??' || printf '')" \
    && if [ -n "$gsl_path" ]; then \
         ln -sfv "$gsl_path" "$(dirname $gsl_path)/libgsl.so.0"; \
    fi \
    && ldconfig \
    && mkdir -p /opt/afni-latest \
    && echo "Downloading AFNI ..." \
    && curl -fL https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C /opt/afni-latest --strip-components 1 \
    && python -m pip install -U img2fmri

# Save specification to JSON.
RUN printf '{ \
  "pkg_manager": "apt", \
  "existing_users": [ \
    "root" \
  ], \
  "instructions": [ \
    { \
      "name": "from_", \
      "kwds": { \
        "base_image": "ubuntu:18.04" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "FSLDIR": "/opt/fsl-6.0.6", \
        "PATH": "/opt/fsl-6.0.6/bin:$PATH", \
        "FSLOUTPUTTYPE": "NIFTI_GZ", \
        "FSLMULTIFILEQUIT": "TRUE", \
        "FSLTCLSH": "/opt/fsl-6.0.6/bin/fsltclsh", \
        "FSLWISH": "/opt/fsl-6.0.6/bin/fslwish", \
        "FSLLOCKDIR": "", \
        "FSLMACHINELIST": "", \
        "FSLREMOTECALL": "", \
        "FSLGECUDAQ": "cuda.q" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "apt-get update -qq\\napt-get install -y -q --no-install-recommends \\\\\\n    bc \\\\\\n    ca-certificates \\\\\\n    curl \\\\\\n    dc \\\\\\n    file \\\\\\n    libfontconfig1 \\\\\\n    libfreetype6 \\\\\\n    libgl1-mesa-dev \\\\\\n    libgl1-mesa-dri \\\\\\n    libglu1-mesa-dev \\\\\\n    libgomp1 \\\\\\n    libice6 \\\\\\n    libopenblas-base \\\\\\n    libxcursor1 \\\\\\n    libxft2 \\\\\\n    libxinerama1 \\\\\\n    libxrandr2 \\\\\\n    libxrender1 \\\\\\n    libxt6 \\\\\\n    nano \\\\\\n    python3 \\\\\\n    sudo \\\\\\n    wget\\nrm -rf /var/lib/apt/lists/*\\n\\necho \\"Installing FSL ...\\"\\ncurl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py | python3 - -d /opt/fsl-6.0.6 -V 6.0.6" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "PATH": "/opt/afni-latest:$PATH", \
        "AFNI_PLUGINPATH": "/opt/afni-latest" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "apt-get update -qq\\napt-get install -y -q --no-install-recommends \\\\\\n    ca-certificates \\\\\\n    cmake \\\\\\n    curl \\\\\\n    ed \\\\\\n    gsl-bin \\\\\\n    libcurl4-openssl-dev \\\\\\n    libgl1-mesa-dri \\\\\\n    libglib2.0-0 \\\\\\n    libglu1-mesa-dev \\\\\\n    libglw1-mesa \\\\\\n    libgomp1 \\\\\\n    libjpeg-turbo8-dev \\\\\\n    libjpeg62 \\\\\\n    libssl-dev \\\\\\n    libudunits2-dev \\\\\\n    libxm4 \\\\\\n    multiarch-support \\\\\\n    netpbm \\\\\\n    python-is-python3 \\\\\\n    python3-pip \\\\\\n    tcsh \\\\\\n    xfonts-base \\\\\\n    xvfb\\nrm -rf /var/lib/apt/lists/*\\n_reproenv_tmppath=\\"$\(mktemp -t tmp.XXXXXXXXXX.deb\)\\"\\ncurl -fsSL --retry 5 -o \\"${_reproenv_tmppath}\\" http://mirrors.kernel.org/debian/pool/main/libx/libxp/libxp6_1.0.2-2_amd64.deb\\napt-get install --yes -q \\"${_reproenv_tmppath}\\"\\nrm \\"${_reproenv_tmppath}\\"\\n_reproenv_tmppath=\\"$\(mktemp -t tmp.XXXXXXXXXX.deb\)\\"\\ncurl -fsSL --retry 5 -o \\"${_reproenv_tmppath}\\" http://snapshot.debian.org/archive/debian-security/20160113T213056Z/pool/updates/main/libp/libpng/libpng12-0_1.2.49-1%%2Bdeb7u2_amd64.deb\\napt-get install --yes -q \\"${_reproenv_tmppath}\\"\\nrm \\"${_reproenv_tmppath}\\"\\napt-get update -qq\\napt-get install --yes --quiet --fix-missing\\nrm -rf /var/lib/apt/lists/*\\n\\ngsl_path=\\"$\(find / -name '"'"'libgsl.so.??'"'"' || printf '"'"''"'"'\)\\"\\nif [ -n \\"$gsl_path\\" ]; then \\\\\\n  ln -sfv \\"$gsl_path\\" \\"$\(dirname $gsl_path\)/libgsl.so.0\\"; \\\\\\nfi\\nldconfig\\nmkdir -p /opt/afni-latest\\necho \\"Downloading AFNI ...\\"\\ncurl -fL https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \\\\\\n| tar -xz -C /opt/afni-latest --strip-components 1" \
      } \
    } \
  ] \
}' > /.reproenv.json
# End saving to specification to JSON.