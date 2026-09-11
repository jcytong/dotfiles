" Managed by dotfiles — language and personal config lives in ~/.vimrc.local

set nocompatible
filetype off
filetype plugin on

" ── Plugins (vim-plug) ──
" Only load if vim-plug is installed
if filereadable(expand('~/.vim/autoload/plug.vim'))
  call plug#begin('~/.vim/plugged')

  " Always load
  Plug 'ojroques/vim-oscyank', {'branch': 'main'}
  Plug 'tomasiser/vim-code-dark'
  Plug 'altercation/vim-colors-solarized'
  Plug 'bling/vim-airline'
  Plug 'tpope/vim-surround'
  Plug 'ctrlpvim/ctrlp.vim'
  Plug 'christoomey/vim-tmux-navigator'
  Plug 'scrooloose/nerdcommenter'
  Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
  Plug 'junegunn/fzf.vim'
  Plug 'andymass/vim-matchup'

  " On-demand loading
  Plug 'godlygeek/tabular', { 'on': 'Tabularize' }
  Plug 'Yggdroot/indentLine', { 'on': 'IndentLinesToggle' }
  Plug 'scrooloose/nerdtree', { 'on':  'NERDTreeToggle' }

  " Language support (lazy-loaded)
  Plug 'pangloss/vim-javascript', { 'for': 'javascript' }
  Plug 'leafgarland/typescript-vim', { 'for': 'typescript' }
  Plug 'tpope/vim-markdown', { 'for': 'markdown' }
  Plug 'posva/vim-vue', { 'for': 'vue' }
  Plug 'elzr/vim-json', { 'for': 'json' }

  " Machine-specific plugins (e.g., Ruby, Clojure, Svelte)
  if filereadable(expand('~/.vimrc.local.plugins'))
    source ~/.vimrc.local.plugins
  endif

  call plug#end()
endif

" ── Core Settings ──
if has('termguicolors')
  let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
  let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
  set termguicolors
endif

set backspace=indent,eol,start
set noswapfile
set t_Co=256
set re=1
set nobackup
set history=50
set ruler
set showcmd
set incsearch
set hlsearch
set expandtab
set tabstop=2
set number
set relativenumber
set cindent
set autoindent
set mouse=
set scrolloff=5
set ignorecase
set smartcase
set hid
set shiftwidth=2
set showmatch
set nowrap
set completeopt=menu,longest,preview
set confirm
set vb t_vb=
set ai
syn on
set synmaxcol=200
let mapleader=","
set background=dark
silent! colorscheme solarized
set encoding=utf-8

" ── fzf runtime path (auto-detect) ──
if isdirectory('/opt/homebrew/opt/fzf')
  set rtp+=/opt/homebrew/opt/fzf
elseif isdirectory('/usr/local/opt/fzf')
  set rtp+=/usr/local/opt/fzf
elseif isdirectory('/usr/share/doc/fzf/examples')
  set rtp+=/usr/share/doc/fzf/examples
elseif isdirectory(expand('~/.fzf'))
  set rtp+=~/.fzf
endif

" ── Clipboard (OSC 52) ──
" Works across SSH/mosh/tmux — yank reaches local clipboard
if exists('##TextYankPost')
  autocmd TextYankPost * if v:event.operator is 'y' && v:event.regname is '' | execute 'OSCYankRegister "' | endif
endif
nmap <leader>y <Plug>OSCYankOperator
vmap <leader>y <Plug>OSCYankVisualSelection

" ── Navigation ──
" Seamless split movement: C-h/j/k/l walks vim's own splits and keeps going
" into the multiplexer's panes once the cursor is against the edge of vim.
"
" The outer half of that — the multiplexer choosing to hand the chord to vim
" rather than acting on it — is configured per multiplexer, because the two do
" it differently:
"   tmux  — if-shell "$is_vim" in ~/.tmux.conf, and vim-tmux-navigator owns
"           the edge case, so in a tmux session just delegate to it.
"   herdr — cannot bind conditionally, so ~/.config/herdr/vim-aware-focus makes
"           the same decision from a [[keys.command]], and the edge case comes
"           back here to s:HerdrFocus().
" Plain vim with neither running keeps the bare <C-w> behaviour these maps
" always had. Note the delegation is new: these four used to be <C-w> maps that
" shadowed vim-tmux-navigator's, so vim never handed focus back under tmux.
" vim-tmux-navigator's plugin/ file is sourced after this vimrc — plugin
" scripts always are — so its own <C-hjkl> maps would otherwise win over the
" ones below, as they quietly did before. Suppressing them leaves the
" :TmuxNavigate* commands in place, which is all the tmux branch needs.
let g:tmux_navigator_no_mappings = 1

function! s:HerdrFocus(direction) abort
  let l:herdr = empty($HERDR_BIN_PATH) ? 'herdr' : $HERDR_BIN_PATH
  let l:cmd = [l:herdr, 'pane', 'focus', '--direction', a:direction,
        \ '--pane', $HERDR_PANE_ID]
  " Fire and forget — the only result is a focus change we never read back,
  " and a synchronous system() would stall the keystroke waiting for it.
  if exists('*jobstart')
    call jobstart(l:cmd)
  elseif exists('*job_start')
    call job_start(l:cmd)
  else
    call system(join(map(copy(l:cmd), 'shellescape(v:val)'), ' '))
  endif
endfunction

function! s:NavigateSplit(wincmd, direction) abort
  if !empty($TMUX) && exists(':TmuxNavigate' . a:direction)
    execute 'TmuxNavigate' . a:direction
    return
  endif
  let l:origin = winnr()
  execute 'wincmd' a:wincmd
  " winnr() unchanged means vim had nowhere left to go in that direction, so
  " the move belongs to the pane next door rather than to this window.
  if winnr() == l:origin && !empty($HERDR_ENV) && !empty($HERDR_PANE_ID)
    call s:HerdrFocus(tolower(a:direction))
  endif
endfunction

function! s:NavigatePrevious() abort
  if !empty($TMUX) && exists(':TmuxNavigatePrevious')
    TmuxNavigatePrevious
    return
  endif
  wincmd p
endfunction

nnoremap <silent> <C-h> :call <SID>NavigateSplit('h', 'Left')<CR>
nnoremap <silent> <C-j> :call <SID>NavigateSplit('j', 'Down')<CR>
nnoremap <silent> <C-k> :call <SID>NavigateSplit('k', 'Up')<CR>
nnoremap <silent> <C-l> :call <SID>NavigateSplit('l', 'Right')<CR>

" tmux: bind-key -n C-\ ... select-pane -l. herdr exposes no last-pane target
" to `pane focus`, so outside tmux this stays inside vim, which is what the
" plugin's own <C-\> map already did here.
nnoremap <silent> <C-\> :call <SID>NavigatePrevious()<CR>

" Fix arrow key maps in tmux
map <Esc>[B <Down>

" Tab navigation
map <S-Right> :tabnext <CR>
map <S-Left> :tabprev <CR>

" Move lines up/down with Alt+j/k (platform-aware)
if has("mac") || has("macunix")
  nnoremap ∆ :m .+1<CR>==
  nnoremap ˚ :m .-2<CR>==
  inoremap ∆ <Esc>:m .+1<CR>==gi
  inoremap ˚ <Esc>:m .-2<CR>==gi
  vnoremap ∆ :m '>+1<CR>gv=gv
  vnoremap ˚ :m '<-2<CR>gv=gv
else
  nnoremap <A-j> :m .+1<CR>==
  nnoremap <A-k> :m .-2<CR>==
  inoremap <A-j> <Esc>:m .+1<CR>==gi
  inoremap <A-k> <Esc>:m .-2<CR>==gi
  vnoremap <A-j> :m '>+1<CR>gv=gv
  vnoremap <A-k> :m '<-2<CR>gv=gv
endif

" ── Search ──
" sudo write
cmap w!! w !sudo tee > /dev/null %

" Prefer ripgrep, fall back to the silver searcher. Both respect .gitignore and
" are fast enough that CtrlP does not need its cache.
"
" --hidden is not optional here: both tools skip dotfiles by default, which in a
" dotfiles repo hides nearly everything (4 files listed instead of 509). .git is
" excluded explicitly since --hidden would otherwise walk into it.
if executable('rg')
  set grepprg=rg\ --vimgrep\ --no-heading\ --smart-case\ --hidden\ --glob\ !.git
  set grepformat=%f:%l:%c:%m
  let g:ctrlp_user_command = 'rg %s --files --color=never --hidden --glob "!.git"'
  let g:ctrlp_use_caching = 0
elseif executable('ag')
  set grepprg=ag\ --nogroup\ --nocolor\ --hidden
  let g:ctrlp_user_command = 'ag %s -l --nocolor --hidden -g ""'
  let g:ctrlp_use_caching = 0
endif

command! -nargs=+ -complete=file -bar Ag silent! grep! <args>|cwindow|redraw!
nnoremap \ :Ag<SPACE>
nnoremap K :grep! "\b<C-R><C-W>\b"<CR>:cw<CR>
nnoremap <Leader>s :%s/\<<C-r><C-w>\>/

" ── Plugin Config ──
let g:airline_powerline_fonts = 1
let g:airline#extensions#tabline#enabled = 1
let g:indentLine_enabled = 0
let g:indentLine_char='¦'

" NERDTree
nmap <silent> <Leader>\ :NERDTreeToggle<CR>
nmap <silent> <Leader>\| :NERDTreeToggle %:p:h<CR>

" NERDCommenter
map <Leader>/ <plug>NERDCommenterToggle<CR>
imap <Leader>/ <Esc><plug>NERDCommenterToggle<CR>i
if has("gui_macvim")
  map <D-/> <plug>NERDCommenterToggle<CR>
  imap <D-/> <Esc><plug>NERDCommenterToggle<CR>i
endif

" ctrlp
nmap <Leader>t :CtrlP<CR>
nmap <Leader>b :CtrlPBuffer<CR>
nmap <Leader>f :CtrlPMixed<CR>

" ── Buffers ──
com! Bdall bufdo bd
com! Bd call Closebufferkeeptab()
com! Be call Closebufferopendir()
com! PrettyJson %!python3 -m json.tool

cmap bdd bn <bar> bd #

map <Leader>bda :Bdall <CR>
map <Leader>dir :echo expand("%:p:h") <CR>
map <Leader>n :bn <CR>
map <Leader>p :bp <CR>
map <leader>q :bp<bar>sp<bar>bn<bar>bd<CR>
map <Leader>o :a<CR><CR>.<CR>
map <Leader>O :i<CR><CR>.<CR>
map <Leader>r "_cw<ESC>p
map <Leader><ESC> :nohlsearch <CR>

" Visual search with //
vnoremap // y/<C-R>"<CR>

" ── Wildignore ──
set wildignore+=*/node_modules/*,dist/*,*/tmp/*,vendor/*,*.class,*.jar

" ── Large File Protection ──
if !exists("my_auto_commands_loaded")
  let my_auto_commands_loaded = 1
  let g:LargeFile = 1024 * 1024 * 10
  augroup LargeFile
    autocmd BufReadPre * let f=expand("<afile>") | if getfsize(f) > g:LargeFile | set eventignore+=FileType | setlocal noswapfile bufhidden=unload buftype=nowrite undolevels=-1 | else | set eventignore-=FileType | endif
  augroup END
endif

" ── Functions ──
function! Gotononamebuffer()
  for i in range(tabpagenr('$'))
    let buflist = tabpagebuflist(i+1)
    for buf in buflist
      if bufname(buf) == "" && bufexists(buf)
        execute 'b' buf
        break
      endif
    endfor
  endfor
  enew
endfunction

function! Closebufferkeeptab()
  let numwin = winnr('$')
  if numwin != 1
    bd
  else
    call Gotononamebuffer()
    bd #
  endif
endfunction

function! Closebufferopendir()
  call Closebufferkeeptab()
  e .
endfunction

" Git pull then refresh all buffers
fun! PullAndRefresh()
  set noconfirm
  !git pull
  bufdo e!
  set confirm
endfun
nmap <Leader>gr call PullAndRefresh()

" ── Language and personal config ──
" Both .local files are tracked in dotfiles so they follow you between machines.
" For a genuinely machine-only override, replace this symlink with a real file:
" the installer skips non-symlinks rather than clobbering them.
if filereadable(expand('~/.vimrc.local'))
  source ~/.vimrc.local
endif
