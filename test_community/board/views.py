from django.shortcuts import render, redirect
from django.core.paginator import Paginator
from django.http import Http404
from fcuser.models import Fcuser
from .models import Board
from .forms import BoardForm

# Create your views here.


def board_detail(request, pk):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    pk = int(pk)
    try:
        board = Board.objects.get(pk=pk)
    except Board.DoesNotExist:
        raise Http404('게시글을 찾을 수 없습니다')

    if board.writer_id == request.session.get('user'):
        context = True
    else:
        context = False

    return render(request, 'board_detail.html', {'board': board, 'writer': context})


def board_write(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    if request.method == 'POST':
        form = BoardForm(request.POST)
        if form.is_valid():
            user_id = request.session.get('user')
            fcuser = Fcuser.objects.get(pk=user_id)

            board = Board()
            title = form.cleaned_data['title']
            board.title = str(title)
            writecontents = form.cleaned_data['writecontents']
            board.writecontents = str(writecontents)
            board.writer = fcuser
            board.save()

            return redirect('/board/list/')
    else:
        form = BoardForm()

    return render(request, 'board_write.html', {'form': form})


def board_list(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    all_boards = Board.objects.all().order_by('-id')
    page = int(request.GET.get('p', 1))
    paginator = Paginator(all_boards, 3)

    boards = paginator.get_page(page)
    return render(request, 'board_list.html', {'boards': boards})


def board_delete(request, pk):
    user = request.session.get('user')
    board = Board.objects.get(pk=pk)

    if request.method == 'GET':
        if board.writer_id == user:
            board.delete()
            return redirect('/board/list')
    else:
        return redirect('/board/detail/')


def board_update(request, pk):
    board = Board.objects.get(pk=pk)

    if request.method == "POST":
        form = BoardForm(request.POST)
        if form.is_valid():
            board = Board.objects.get(pk=pk)
            board.title = str(form.cleaned_data['title'])
            writecontents = form.cleaned_data['writecontents']
            board.writecontents = str(writecontents)
            board.save()

        return redirect('/board/list')

    else:
        form = BoardForm()

    return render(request, 'board_write.html', {'form': form})
