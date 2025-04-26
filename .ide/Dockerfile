# .ide/Dockerfile

FROM ccmagruder/devcontainer-symmetry:latest

RUN apt update -y && apt upgrade -y \
  && apt install -y --no-install-recommends \
    bat \
    clangd \
    curl \
    fd-find \
    gem \
    language-pack-en \
    less \
    lua5.1 \
    luarocks \
    nodejs \
    npm \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    ripgrep \
    ruby \
    ruby-dev \
    tmux \
    tree-sitter-cli \
    zsh \
  && rm -rf /var/lib/apt/lists/* \
  # Configure shell
  && chsh -s /usr/bin/zsh \
  && ln -s /usr/bin/batcat /usr/local/bin/bat \
  && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
  && sed -i -e 's/robbyrussell/agnoster/' /root/.zshrc \
  && echo "cd /root" >> /root/.zshrc \
  # Install neovim
  && curl -O -L https://github.com/neovim/neovim/releases/download/nightly/nvim-linux-x86_64.tar.gz \
  && tar xzvf nvim-linux-x86_64.tar.gz -C /usr/local/ --strip-components=1 \
  && rm nvim-linux-x86_64.tar.gz \
  # Install tmux plugin manager
  && git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm \
  && echo "if [ ! -f /root/.tmux/tpm_loaded ]; then" >> ~/.zshrc \
  && echo "    ~/.tmux/plugins/tpm/scripts/install_plugins.sh \\" >> ~/.zshrc \
  && echo "        && touch /root/.tmux/tpm_loaded" >> ~/.zshrc \
  && echo "fi" >> ~/.zshrc \
  # Install yaml-language-server
  && npm install -g yaml-language-server \
  # Configure python provider
  && python3 -m venv /.venv \
  && PATH=/.venv/bin:$PATH pip install jupyter-client neovim pyright\
  && echo "source /.venv/bin/activate" >> ~/.zshrc \
  # Configure node provider
  && npm install -g neovim \
  # Configure perl provider
  && cpan install App::cpanminus \
  && cpanm -n Neovim::Ext \
  # Configure ruby provider
  && gem install neovim \
  # Configure gpg credentials
  && echo "export GPG_TTY=\$(tty)" >> ~/.zshrc \
  && echo "if [ ! -d /root/.gnupg ]; then" >> ~/.zshrc \
  && echo "    gpg --batch --import /root/.private.key" >> ~/.zshrc \
  && echo "fi" >> ~/.zshrc \
  && echo "alias ll=\"ls -alh\"" >> ~/.zshrc \
  && echo "alias gs=\"git status\"" >> ~/.zshrc \
  && echo "cd /root/gemm" >> ~/.zshrc
