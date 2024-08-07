curl https://pyenv.run | bash
echo '' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo '' >> ~/.profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile

apt-get update
apt-get upgrade -y
apt-get install libffi-dev

LDFLAGS="-Wl,-rpath,/opt/conda/lib" CONFIGURE_OPTS="--with-openssl=/opt/conda" pyenv install -v 3.10.9

pip install pipenv
# pipenv lock
# pipenv sync