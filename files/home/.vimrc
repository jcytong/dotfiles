set nocompatible
filetype off            "required by vundle
filetype plugin on

call plug#begin('~/.vim/plugged')

" My Plugins here:
 "
 " original repos on github
 "
 " These are interesting that I want to try someday
 "Plug 'danro/rename.vim'
 "Plug 'ervandew/supertab'
 
 " Always load
 "Plug 'neoclide/coc.nvim', {'branch': 'release'}
 Plug 'tomasiser/vim-code-dark'
 Plug 'altercation/vim-colors-solarized'
 Plug 'bling/vim-airline'
 Plug 'tpope/vim-surround'
 "Plug 'wincent/Command-T'
 Plug 'ctrlpvim/ctrlp.vim'
 Plug 'scrooloose/syntastic'
 Plug 'christoomey/vim-tmux-navigator'
 Plug 'scrooloose/nerdcommenter'
 Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
 Plug 'junegunn/fzf.vim'
 Plug 'andymass/vim-matchup'

 "" On-demand loading
 Plug 'godlygeek/tabular', { 'on': 'Tabularize' }
 Plug 'Yggdroot/indentLine', { 'on': 'IndentLinesToggle' }
 Plug 'scrooloose/nerdtree', { 'on':  'NERDTreeToggle' }

 " Clojure
 Plug 'tpope/vim-fireplace', { 'for': 'clojure' }
 Plug 'venantius/vim-cljfmt', { 'for': 'clojure' }
 Plug 'guns/vim-clojure-static', { 'for': 'clojure' }
 Plug 'guns/vim-sexp', { 'for': 'clojure' }
 Plug 'tpope/vim-sexp-mappings-for-regular-people', { 'for': 'clojure' }
 "Plug 'junegunn/rainbow_parentheses.vim', { 'for': 'clojure' }
 Plug 'luochen1990/rainbow', { 'for': 'clojure' }
 Plug 'kovisoft/paredit', { 'for': ['scheme'] }

 " Ruby
 Plug 'thoughtbot/vim-rspec', { 'for': 'ruby' }
 Plug 'vim-ruby/vim-ruby', { 'for': 'ruby' }
 Plug 'tpope/vim-rails', { 'for': 'ruby' }
 Plug 'tpope/vim-endwise', { 'for': 'ruby' }
 Plug 'vim-scripts/ruby-matchit', { 'for': 'ruby' }

 " Javascript
 Plug 'pangloss/vim-javascript', { 'for': 'javascript' }

 " Typescript
 Plug 'leafgarland/typescript-vim', { 'for': 'typescript' }

 " Web
 Plug 'kchmck/vim-coffee-script', { 'for': 'coffeescript' }
 Plug 'gregsexton/MatchTag', { 'for': 'html' }
 Plug 'tpope/vim-markdown', { 'for': 'markdown' }
 Plug 'cakebaker/scss-syntax.vim', { 'for': ['sass', 'scss'] }
 Plug 'posva/vim-vue', { 'for': 'vue' }
 Plug 'evanleck/vim-svelte', { 'for': 'svelte' }
 Plug 'elzr/vim-json', { 'for': 'json' }

 " GraphQL
 Plug 'jparise/vim-graphql', { 'for': 'graphql' }

 " Snipmate looks cool and depends on vim-addon-mw-utils and tlib_vim
 "Plug 'MarcWeber/vim-addon-mw-utils'
 "Plug 'tomtom/tlib_vim'
 "Plug 'garbas/vim-snipmate'
 "Plug 'honza/vim-snippets'

 " Autocomplete and type hint for js
 "Plug 'marijnh/tern_for_vim'
 "Plug 'rstacruz/sparkup', {'rtp': 'vim/'}
 "Plug 'tristen/vim-sparkup'
 "Plug 'tpope/vim-fugitive'
 "Plug 'terryma/vim-multiple-cursors'
 
 " These plugin seems slow
 "Plug 'mileszs/ack.vim'
 "Plug 'rizzatti/funcoo.vim'
 "Plug 'rizzatti/dash.vim'

 "Plug 'vim-scripts/greplace.vim'

 "Plug 'Lokaltog/vim-easymotion'
 
 " vim-scripts repos
 "Plug 'L9'
 "Plug 'FuzzyFinder'

call plug#end()

if has('termguicolors')
  let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
  let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
  set termguicolors
endif

" allow backspacing over everything in insert mode
set backspace=indent,eol,start
set noswapfile
set t_Co=256
set re=1                " Fix slow scrolling when filetype=ruby and set relativenumber - https://stackoverflow.com/a/16920294/
set nobackup            " DON'T keep a backup file
set history=50          " keep 50 lines of command line history
set ruler               " show the cursor position all the time
set showcmd             " display incomplete commands
set incsearch           " do incremental searching
set hlsearch            " hilight search
set expandtab           " insert spaces instead of tabs
set tabstop=2
set number              " line numbers
set relativenumber      " show relative line number
set cindent
set autoindent
"set mouse=a             " use mouse in xterm to scroll
set mouse=              " disable mouse in xterm so you can copy without using the option key
set scrolloff=5         " 5 lines bevore and after the current line when scrolling
set ignorecase          " ignore case
set smartcase           " but don't ignore it, when search string contains uppercase letters
set hid                 " allow switching buffers, which have unsaved changes
set shiftwidth=2        " 2 characters for indenting
set showmatch           " showmatch: Show the matching bracket for the last ')'?
set nowrap              " don't wrap by default
set completeopt=menu,longest,preview
set confirm
set vb t_vb=            " disable beep
set ai
syn on
set synmaxcol=200
let mapleader=","
set background=dark
colorscheme solarized
set encoding=utf-8
if has("mac") || has("macunix")
  set lines=49
  set columns=90
  set guifont=Hack:h7
  let g:Powerline_symbols = 'fancy'
  set fillchars+=stl:\ ,stlnc:\
  if !has('nvim')
    set term=xterm-256color
    set termencoding=utf-8
  endif
endif

" fzf runtime path (auto-detect)
if isdirectory('/opt/homebrew/opt/fzf')
  set rtp+=/opt/homebrew/opt/fzf
elseif isdirectory('/usr/local/opt/fzf')
  set rtp+=/usr/local/opt/fzf
elseif isdirectory('/usr/share/doc/fzf/examples')
  set rtp+=/usr/share/doc/fzf/examples
elseif isdirectory(expand('~/.fzf'))
  set rtp+=~/.fzf
endif

" CoC extensions
let g:coc_global_extensions = ['coc-tsserver']
" Remap keys for applying codeAction to the current line.
nmap <leader>ac  <Plug>(coc-codeaction)
" Apply AutoFix to problem on the current line.
nmap <leader>qf  <Plug>(coc-fix-current)
" GoTo code navigation.
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)

" https://stackoverflow.com/questions/2600783/how-does-the-vim-write-with-sudo-trick-work
cmap w!! w !sudo tee > /dev/null %

"cmap bdd bn | bd #

" https://robots.thoughtbot.com/faster-grepping-in-vim
" The Silver Searcher
if executable('ag')
  " Use ag over grep
  set grepprg=ag\ --nogroup\ --nocolor

  " Use ag in CtrlP for listing files. Lightning fast and respects
  " .gitignore
  let g:ctrlp_user_command = 'ag %s -l --nocolor -g ""'

  " ag is fast enough that CtrlP doesn't need to cache
  let g:ctrlp_use_caching = 0
endif

" bind \ (backward slash) to grep shortcut
command! -nargs=+ -complete=file -bar Ag silent! grep! <args>|cwindow|redraw!
nnoremap \ :Ag<SPACE>

" quick search replace all under cursor
nnoremap <Leader>s :%s/\<<C-r><C-w>\>/

let g:airline_powerline_fonts = 1
" Enable https://github.com/luochen1990/rainbow plugin
let g:rainbow_active = 1 "0 if you want to enable it later via :RainbowToggle

" Protect large files from sourcing and other overhead.
" Files become read only
if !exists("my_auto_commands_loaded")
  let my_auto_commands_loaded = 1
  " Large files are > 10M
  " Set options:
  " eventignore+=FileType (no syntax highlighting etc
  " assumes FileType always on)
  " noswapfile (save copy of file)
  " bufhidden=unload (save memory when other file is viewed)
  " buftype=nowritefile (is read-only)
  " undolevels=-1 (no undo possible)
  let g:LargeFile = 1024 * 1024 * 10
  augroup LargeFile
  let g:LargeFile = 1024 * 1024 * 10
  augroup LargeFile
    autocmd BufReadPre * let f=expand("<afile>") | if getfsize(f) > g:LargeFile | set eventignore+=FileType | setlocal noswapfile bufhidden=unload buftype=nowrite undolevels=-1 | else | set eventignore-=FileType | endif
    augroup END
  endif
""""""""""""""""""""""""""""""""""" Commands """""""""""""""""""""""""""""""""""
" Delete all buffers
com! Bdall bufdo bd
"com! Bd call Closebufferkeeptab()
com! Be call Closebufferopendir()
com! PrettyJson %!python3 -m json.tool

""""""""""""""""""""""""""""""""""" Mappings """""""""""""""""""""""""""""""""""
" bind K to grep word under cursor
nnoremap K :grep! "\b<C-R><C-W>\b"<CR>:cw<CR>

"MOVEMENT
"--------
" Fix weird error when arrow key maps incorrectly when using vim with tmux
" http://superuser.com/questions/237751/messed-up-keys-in-vim-when-running-inside-tmux
map <Esc>[B <Down>

" Go to previous/next tab with Shift-left/right arrow respectively
map <S-Right> :tabnext <CR>
map <S-Left> :tabprev <CR>
" Map Ctrl-j,k for page up/down
"nmap <C-J> <C-F>
"nmap <C-K> <C-B>

" Move lines up/down with Alt+j/k
if has("mac") || has("macunix")
  " macOS Terminal sends special chars for Option+j/k
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

" Shortcut to rapidly toggle `set list`
"nmap <leader>l :set list!<CR>

" Use the same symbols as TextMate for tabstops and EOLs
"set listchars=tab:▸\ ,eol:¬

"BUFFERS
"-------
" Delete all buffers (Maps Bdall)
map <Leader>bda :Bdall <CR>
" Show current directory
map <Leader>dir :echo expand("%:p:h") <CR>
" next buffer
map <Leader>n :bn <CR>
map <Leader>p :bp <CR>
" Close buffer without closing window
map <leader>q :bp<bar>sp<bar>bn<bar>bd<CR>

"EDIT
"----
" Insert a line below the current line
map <Leader>o :a<CR><CR>.<CR>
" Insert a line above the current line
map <Leader>O :i<CR><CR>.<CR>

" Replace word under cursor with default register
"map <Leader>r "_cw<ESC>p

map <Leader><ESC> :nohlsearch <CR>

" The :vnoremap command maps // in visual mode to run the commands y/<C-R>"<CR>
" which copies the visually selected text, then starts a search command and
" pastes the copied text into the search.
vnoremap // y/<C-R>"<CR>

" Git pull then refresh all buffers
fun! PullAndRefresh()
  set noconfirm
  !git pull
  bufdo e!
  set confirm
endfun

nmap <Leader>gr call PullAndRefresh()

"PLUGINS
"-------
"IndentLines
let g:indentLine_enabled = 0
let g:indentLine_char='¦'

"NERDTree
nmap <silent> <Leader>\ :NERDTreeToggle<CR>
nmap <silent> <Leader>\| :NERDTreeToggle %:p:h<CR>

" NERDCommenter Command-/ to toggle comments (macOS GUI only)
if has("gui_macvim")
  map <D-/> <plug>NERDCommenterToggle<CR>
  imap <D-/> <Esc><plug>NERDCommenterToggle<CR>i
endif
map <Leader>/ <plug>NERDCommenterToggle<CR>
imap <Leader>/ <Esc><plug>NERDCommenterToggle<CR>i

" Command-T plugin
" Ignore angular dir
" Ignore rails
set wildignore+=*/node_modules/*,app/images/*,dist/*,test/unit/coverage/*,karma_html/*,node_modules/*,*/tmp/*,vendor/*,*.class,*.jar,*/_site,*/__sapper__

" Only list files under the pwd directory
"let g:CommandTMaxDepth=10
"let g:CommandTTraverseSCM="pwd"
"let g:CommandTScanDotDirectories=".tmp/**"
" Cancel listing with Esc
"let g:CommandTCancelMap=['<ESC>','<C-c>']

" It triggers CommandTFlush whenever a file is written and also whenever Vim's
" window gains focus. This is useful when you create files outside of vim - for
" example by switching between branches in your version control system. The new
" files will be available in CommandT immediately after you re-enter Vim.
"
" SO Question & Answer
" http://stackoverflow.com/questions/3486747/run-the-commandtflush-command-when-a-new-file-is-written
" http://stackoverflow.com/a/5791719/573486
"augroup CommandTExtension
  "autocmd!
  "autocmd FocusGained * CommandTFlush
  "autocmd BufWritePost * CommandTFlush
"augroup END

" ctrlp plugin
nmap <Leader>t :CtrlP<CR>
nmap <Leader>b :CtrlPBuffer<CR>
nmap <Leader>f :CtrlPMixed<CR>

" vim-coffee-script plugin
" https://github.com/kchmck/vim-coffee-script#two-space-indentation
" https://github.com/kchmck/vim-coffee-script#fold-by-indentation
autocmd BufNewFile,BufReadPost *.coffee setl shiftwidth=2 expandtab
autocmd BufNewFile,BufReadPost *.coffee setl foldmethod=indent nofoldenable 

" dash.vim (macOS only)
" https://github.com/rizzatti/dash.vim
if has("mac") || has("macunix")
  map <Leader>d :call Dash<CR>
endif

" vim-rspec
" https://github.com/thoughtbot/vim-rspec/blob/master/README.md

" Remove PGGSSENCMODE=\"disable\" when it's fixed
" https://github.com/ged/ruby-pg/issues/538#issuecomment-1591629049
let g:rspec_command = "!PGGSSENCMODE=\"disable\" bin/rspec --drb {spec}"

map <Leader>r :call RunCurrentSpecFile()<CR>
map <Leader>m :call RunNearestSpec()<CR>
map <Leader>l :call RunLastSpec()<CR>
"map <Leader>a :call RunAllSpecs()<CR>

" vim-airline
" https://github.com/bling/vim-airline#straightforward-customization
let g:airline#extensions#tabline#enabled = 1
""""""""""""""""""""""""""""""""" Autocommands """""""""""""""""""""""""""""""""
" Change the directory where buffer is located
" Temporarily disable this to test out Command-T plugin
"autocmd BufEnter * cd %:p:h
autocmd FileType javascript set omnifunc=javascriptcomplete#CompleteJS
autocmd FileType html set omnifunc=htmlcomplete#CompleteTags
autocmd FileType css set omnifunc=csscomplete#CompleteCSS

" HTML (tab width 2 chr, no wrapping)
autocmd FileType html set sw=2
autocmd FileType html set ts=2
autocmd FileType html set sts=2
autocmd FileType html set textwidth=0
" XHTML (tab width 2 chr, no wrapping)
autocmd FileType xhtml set sw=2
autocmd FileType xhtml set ts=2
autocmd FileType xhtml set sts=2
autocmd FileType xhtml set textwidth=0
" CSS (tab width 2 chr, wrap at 79th char)
autocmd FileType css set sw=2
autocmd FileType css set ts=2
autocmd FileType css set sts=2

"""""""""""""""""""""""""""""""""" Functions """""""""""""""""""""""""""""""""""
set diffexpr=MyDiff()
function! MyDiff()
    let opt = ""
    if &diffopt =~ "icase"
     let opt = opt . "-i "
    endif
    if &diffopt =~ "iwhite"
     let opt = opt . "-b "
    endif

    silent execute "!/usr/bin/diff -a " . opt . v:fname_in
    \ . " " . v:fname_new . " > " . v:fname_out
endfunction

set patchexpr=MyPatch()
function! MyPatch()
    silent execute "!/usr/bin/patch -o " . v:fname_out . " " . v:fname_in . " < " . v:fname_diff
endfunction

function! Gotononamebuffer()
    for i in range(tabpagenr('$'))
        buflist = tabpagebuflist(i+1)
        for buf in buflist
            if bufname(buf) == "" && bufexists(buf)
                b buf
                break
            endif
        endfor
    endfor
    enew
endfunction

" Closing the buffer while keeping tab
function! Closebufferkeeptab()
    let numwin = winnr('$')

    " More than one window
    if numwin != 1
        bd
    else
        call Gotononamebuffer()
        bd #
    endif 
endfunction

" Opens current directory view then deletes the last buffer
function! Closebufferopendir()
    call Closebufferkeeptab()
    e .
endfunction
"------------------------------------------------------------------------------"
"                              JavaScript DEVELOPMENT                          "
"------------------------------------------------------------------------------"
" autoexpanding abbreviations for insert mode
"iab it(
"it("", function() {
"});
"------------------------------------------------------------------------------"
"                     END OF SECTION FOR JavaScript DEVELOPMENT                "
"------------------------------------------------------------------------------"
"
"------------------------------------------------------------------------------"
"                              JAVA DEVELOPMENT                                "
"------------------------------------------------------------------------------"
""""""""""""""""""""""""""""""" Code Completion """"""""""""""""""""""""""""""""
set complete=],.,b,w,t,k
set dictionary=~/.vimKeywords

" Set up tab for code completion
inoremap  =InsertTabWrapper ("forward")
inoremap =InsertTabWrapper ("backward")

function! InsertTabWrapper(direction)
  let col = col('.') - 1
  if !col || getline('.')[col - 1] !~ '\k'
    return "\"
  elseif "backward" == a:direction
    return "\"
  else
    return "\"
  endif
endfunction
"------------------------------------------------------------------------------"
"                     END OF SECTION FOR JAVA DEVELOPEMNT                      "
"------------------------------------------------------------------------------"
