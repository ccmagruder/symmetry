#include <ncurses.h>
#include <iostream>
#include <sstream>
// #include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include "pbar.h"

int main()
{
    clock_t _tbegin = time(0);
    initscr();
    noecho();
    std::stringstream ss;

    WINDOW *border = newwin(12,80,0,0);
    WINDOW *win = newwin(10,78,1,1);
    wborder(border, 0, 0, 0, 0, 0, 0, 0, 0);
    wrefresh(border);
    wrefresh(win);
    pbar i(100,40,"tag",ss);
    for (i=1;i<uint64_t(100);i++)
    {
        std::string text = ss.str();
        mvwprintw(win,0,0,"%s",text.c_str());
        mvwprintw(win,1,0,"%s",std::to_string(int(i)).c_str());
            // mvwprintw(win,2,0,);
        wrefresh(win);
            _tbegin = clock();
            while (clock() - _tbegin < 2e5){}
    }
    getchar();
    endwin();
    // std::cout << ss.str();

    return 0;
}