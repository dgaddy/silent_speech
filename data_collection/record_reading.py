import sys
import os
import textwrap
import curses
import soundfile as sf
import json
import numpy as np

from read_book import Book
from record_data import Recorder

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('output_directory', None, 'where to save outputs')
flags.DEFINE_string('book_file', None, 'text to read')
flags.mark_flag_as_required('output_directory')
flags.mark_flag_as_required('book_file')

def display_sentence(sentence, win):
    height, width = win.getmaxyx()
    win.clear()
    wrapped_sentence = textwrap.wrap(sentence, width)
    for i, text in enumerate(wrapped_sentence):
        if i >= height:
            break
        win.addstr(i, 0, text)
    win.refresh()

def save_data(output_idx, data, book):
    emg, audio, button, chunk_info = data
    emg_file = os.path.join(FLAGS.output_directory, f'{output_idx}_emg.npy')
    audio_file = os.path.join(FLAGS.output_directory, f'{output_idx}_audio.flac')
    button_file = os.path.join(FLAGS.output_directory, f'{output_idx}_button.npy')
    info_file = os.path.join(FLAGS.output_directory, f'{output_idx}_info.json')
    assert not os.path.exists(emg_file), 'trying to overwrite existing file'
    np.save(emg_file, emg)
    sf.write(audio_file, audio, 16000)
    np.save(button_file, button)

    if book is None:
        # special silence segment
        bf = ''
        bi = -1
        t = ''
    else:
        bf = book.file
        bi = book.current_index
        t = book.current_sentence()

    with open(info_file, 'w') as f:
        json.dump({'book':bf, 'sentence_index':bi, 'text':t, 'chunks':chunk_info}, f)


def get_ends(data):
    emg, audio, button, chunk_info = data
    emg_start = emg[:500,:]
    emg_end = emg[-500:,:]
    dummy_audio = np.zeros(8000)
    dummy_button = np.zeros(500, dtype=np.bool)
    chunk_info = [(500,8000,500)]
    return (emg_start, dummy_audio, dummy_button, chunk_info), (emg_end, dummy_audio, dummy_button, chunk_info)

def main(stdscr):
    os.makedirs(FLAGS.output_directory, exist_ok=False)
    output_idx = 0

    curses.curs_set(False)
    stdscr.nodelay(True)

    text_win = curses.newwin(curses.LINES-1, curses.COLS, 0, 0)

    recording = False

    with Recorder(debug=FLAGS.debug) as r, Book(FLAGS.book_file) as book:
        stdscr.clear()
        stdscr.addstr(0,0,'<Press any key to begin.>')
        stdscr.refresh()

        while True:
            r.update()
            if not recording:
                c = stdscr.getch()
                if c >= 0:
                    # keypress
                    recording = True
                    r.get_data() # clear data
                    stdscr.addstr(curses.LINES-1, 0, "Type 'q' to quit, 'n' or ' ' for next, 'r' to restart segment")
                    display_sentence('<silence>', text_win)
                    stdscr.refresh()
            else:
                c = stdscr.getch()
                if c < 0:
                    # no keypress
                    pass
                elif c == ord('q'):
                    start_data, end_data = get_ends(r.get_data())
                    save_data(output_idx, start_data, None)
                    break
                elif c == ord('n') or c == ord(' '):
                    data = r.get_data()

                    if output_idx == 0:
                        save_data(output_idx, data, None)
                    else:
                        save_data(output_idx, data, book)
                        book.next()

                    output_idx += 1
                    display_sentence(book.current_sentence(), text_win)
                elif c == ord('r'):
                    if output_idx == 0:
                        r.get_data() # clear data
                    else:
                        start_data, end_data = get_ends(r.get_data())
                        save_data(output_idx, start_data, None)
                        output_idx += 1
                        save_data(output_idx, end_data, None)
                        output_idx += 1


FLAGS(sys.argv)
curses.wrapper(main)
