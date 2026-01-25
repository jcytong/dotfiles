# dotfiles

Personal dotfiles managed with symlinks.

## Setup

```bash
git clone git@github.com:jcytong/dotfiles.git ~/.dotfiles
cd ~/.dotfiles
./bin/dotf link
```

## Adding Dotfiles

Put files in `files/home/` mirroring your home directory structure:

```
files/home/.gitconfig        → ~/.gitconfig
files/home/.config/nvim/     → ~/.config/nvim/
files/home/.zshrc            → ~/.zshrc
```

Then run `./bin/dotf link` to create symlinks.

## Structure

```
bin/dotf       Symlink script
files/home/    Dotfiles (mirrors ~/)
```
