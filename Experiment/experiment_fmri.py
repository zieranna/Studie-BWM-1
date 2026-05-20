from statistics import mean
import os
import math
import time
import shutil
from random import sample
import numpy as np
import pandas as pd
import scipy.io
from psychopy.hardware import keyboard
from psychopy.misc import fromFile
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)

import deadDots

from psychopy import gui, visual, core, data, monitors, event


# ====================== 1. Set up the experiment ======================

# ================================================================================
# Monitor parameters — set `window` to select your setup
# Each block configures: gammaGrid, monitor name, distance, size, resolution

window = "___"  # e.g. "mrt" or "fuijutsu"

if window == "mrt":
    gammaGrid = ___  # e.g. [[min, max, gamma, "nan", "nan", "nan"], ...]
    mon = monitors.Monitor('___')
    mon.setDistance(___)   # cm
    mon.setWidth(___)      # cm
    mon.setSizePix([___, ___])
    mon.setGammaGrid(gammaGrid)
    mon.setLineariseMethod(1)
    win = visual.Window(
        size=(___, ___), fullscr=False, screen=0,
        winType='pyglet', allowGUI=True, allowStencil=True,
        monitor=mon, color=[50, 50, 50], colorSpace='rgb255',
        blendMode='avg', useFBO=False, units='deg', multiSample=True, numSamples=16, waitBlanking=False)

if window == "fuijutsu":
    gammaGrid = ___  # e.g. [[min, max, gamma, "nan", "nan", "nan"], ...]
    mon = monitors.Monitor('___')
    mon.setDistance(___)   # cm
    mon.setWidth(___)      # cm
    mon.setSizePix([___, ___])
    mon.setGammaGrid(gammaGrid)
    mon.setLineariseMethod(1)
    win = visual.Window(
        size=(___, ___), fullscr=False, screen=0,
        winType='pyglet', allowGUI=True, allowStencil=True,
        monitor=mon, color=[50, 50, 50], colorSpace='rgb255',
        blendMode='avg', useFBO=False, units='deg', multiSample=True, numSamples=16, waitBlanking=False)

# ================================================================================
# Store info about the experiment session
psychopyVersion = '2022.1.3'
expName = 'Binding Working Memory Experiment'
expInfo = {'Participant': '1', 'Session N°': '1', 'RoundStart': '1', 'Run': 'all'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()
expInfo['date'] = data.getDateStr()
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

win.fullscr = True

# ================================================================================
# Directory structure:
#   <parent_folder>/
#     sub-<ID>/
#       Input/
#         beh/
#           conditions_<session>.xlsx     ← trial conditions per session
#       Output/
#         beh/
#           BindingExp/
#             <session>/                  ← trial results (xlsx, hdf5), 5-press logs
#         behQuest/                       ← staircase .psydat and .xlsx (persist across sessions)
#     Look_up_table.xlsx                  ← shared color look-up table
#     ColorWheel360.mat                   ← shared color wheel matrix
#     mouse_cursor_black-modified.png     ← cursor image for response screen

parent_folder = "___"  # e.g. "/home/user/experiments/Studie1-BWM-1"
fn_stimuli = parent_folder  # shared stimuli live directly in parent_folder

fn_subject = f"{parent_folder}/sub-{expInfo['Participant']}"
fn_results = f"{fn_subject}/Output/beh/BindingExp/{expInfo['Session N°']}"
fn_quest = f"{fn_subject}/Output/behQuest"
[os.makedirs(fn) if not os.path.exists(fn) else print(f" {fn} Folder already exists") for fn in (fn_results, fn_quest)]

fn_condition = f"{fn_subject}/Input/beh/conditions_{expInfo['Session N°']}.xlsx"
conditions = pd.read_excel(fn_condition)
shutil.copy(fn_condition, f"{fn_results}/Input_conditions.xlsx")

# Output file names
fn_hdf5 = fn_results + '/binding_exp_results.hdf5'
fn_excel = f"{fn_results}/binding_exp_results.xlsx"
fn_excel_lst5 = f"{fn_results}/tr_5press.xlsx"
fn_hdf5_lst5 = f"{fn_results}/tr_5press.hdf5"

fn_excel_bac = f"{fn_results}/binding_exp_bac_{expInfo['date']}.xlsx"
fn_hdf5_bac = f"{fn_results}/binding_exp_bac_{expInfo['date']}.hdf5"
fn_excel_5_bac = f"{fn_results}/tr_5press_bac_{expInfo['date']}.xlsx"
fn_hdf5_5_bac = f"{fn_results}/tr_5press_bac_{expInfo['date']}.hdf5"

[os.rename(fn, fn_bac) for fn, fn_bac in zip([fn_excel, fn_hdf5, fn_excel_lst5, fn_hdf5_lst5], [fn_excel_bac, fn_hdf5_bac, fn_excel_5_bac, fn_hdf5_5_bac]) if os.path.exists(fn)]

# Load stimulus look-up tables
colortriangle = pd.read_excel(f"{fn_stimuli}/Look_up_table.xlsx")
col_matrix = scipy.io.loadmat(f'{fn_stimuli}/ColorWheel360.mat')

endExpNow = False
frameTolerance = 0.001

expInfo['frameRate'] = win.getActualFrameRate()
print(f"Frame Rate: {expInfo['frameRate']}")
if expInfo['frameRate'] == None:
    raise ValueError("Frame Rate was not measured")
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.256

ioSession = ioServer = eyetracker = None

defaultKeyboard = keyboard.Keyboard(backend='event')
keys = keyboard.Keyboard()

if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
    core.quit()

# MRI color column names differ from behavioral version (R_mri, G_mri, B_mri)
colortriangle = colortriangle[["color", "idx", "R_mri", "G_mri", "B_mri"]]
colortriangle = colortriangle.rename(columns={"R_mri": "R", "G_mri": "G", "B_mri": "B"})
colortriangle_rgb = colortriangle[["R", "G", "B"]]


# ====================== 2. Set up the functions for sequences ======================

def check_for_keys(key="5"):
    if keys.waitKeys(keyList=[key]):
        return True
    else:
        return False


def feedback(df, block):
    """
    Prints per-error-type breakdown for the given block(s).
    error codes: 0 = correct, 1 = feature error, 2 = binding error, 3 = no response
    """
    if type(block) == int:
        df = df[df["Round"] == block]
    elif type(block) == list:
        df = df[df["Round"].isin(block)]

    error = df["Error"]
    correct = round(error[error == 0].count() / len(error), 2) * 100
    feature = round(error[error == 1].count() / len(error), 2) * 100
    binding = round(error[error == 2].count() / len(error), 2) * 100
    no_response = round(error[error == 3].count() / len(error), 2) * 100
    print(f"Ergebnisse Block {block}: \n \ncorrect: {correct} %, \nfeature: {feature} %, \nbinding: {binding} %, \nno response: {no_response} %")


def time_to_frame(t, FR):
    """Convert time (in seconds) to frame index (zero-indexed).

    Args:
        t: Time in seconds (start time of the frame).
        FR: frame duration in seconds.

    Returns:
        frame_idx: Frame index (non-negative int).
    """
    FR = FR * 1000
    if t < 0:
        raise ValueError("Time should be positive!")
    frame_idx = math.floor(float(t) * 1000 / float(FR))
    frame_idx = int(round(frame_idx))
    assert frame_idx >= 0, "Frame index should be non-negative."
    return frame_idx


def route_txt_cent(txt, col, win, defaultKeyboard, timer):
    """
    Displays a short centered text for a fixed duration (e.g. round announcement).
    Includes a 'please wait' message below; does not wait for key press.
    Returns onset time.
    """
    text = visual.TextBox2(win=win, text=txt, size=(30, 4), letterHeight=1,
                           color=col, colorSpace='rgb255', pos=(0, 4), units='deg', bold=True,
                           alignment='center')
    text_enter = visual.TextBox2(win=win, text='Gleich geht es weiter, \nWarten Sie ab und bleiben Sie ruhig liegen',
                                 size=(30, 4), letterHeight=0.8,
                                 color=col, colorSpace='rgb255', pos=(0, -6), units='deg', alignment='center')
    frames = time_to_frame(timer, frameDur)
    t = time.time()
    for frameN in range(frames):
        text.draw()
        text_enter.draw()
        win.flip()
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
    return t


def route_fix(win, defaultKeyboard, col, timer=None, dot_size=(0.25, 0.25)):
    """
    Creates a fixation point.

    If timer is None: returns the fixation stimulus for use in another routine.
    If timer is given: displays fixation for that duration and returns the onset time.
    """
    fix = visual.ShapeStim(win=win, name='Fixation', vertices='circle', units='deg', size=dot_size, pos=(0, 0),
                           ori=0.0, fillColor=col, lineColor=None, colorSpace='rgb255', interpolate=True)

    fix_onset = time.time()
    if timer:
        frames = time_to_frame(timer, frameDur)
        for frameN in range(frames):
            fix.draw()
            win.flip()
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()

    return fix_onset if timer else fix


def motion_series1(win, defaultKeyboard, dir1, col1, fix="normal", timer=0.5):
    fix = route_fix(win, defaultKeyboard, [198, 198, 198])
    frames = time_to_frame(timer, frameDur)
    stim1 = deadDots.zombieDots(win=win, fieldSize=15, units="deg", nDots=300, dotLife=-1, fieldPos=(0, 0),
                                fieldShape='circle', dir=dir1, coherence=1, dotSize=9, colorSpace='rgb255',
                                color=col1, speed=0.25, signalDots='different', noiseDots='direction')
    t = time.time()
    for frameN in range(frames):
        fix.draw()
        stim1.draw()
        win.flip()
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
    return t


def motion_mask(win, defaultKeyboard, timer):
    fix = route_fix(win, defaultKeyboard, [198, 198, 198])
    frames = time_to_frame(timer, frameDur)
    rd_lst = (np.arange(start=1, stop=360, step=20)).tolist()
    rd_col = sample(rd_lst, 6)
    rd_mot = sample(rd_lst, 6)

    stims = []
    for i in range(6):
        stims.append(visual.DotStim(win=win, fieldSize=15, units="deg", nDots=35, dotLife=-1, fieldPos=(0, 0),
                                    fieldShape='circle', dir=rd_mot[i], coherence=0, dotSize=9, colorSpace='rgb255',
                                    color=col_matrix["fullcolormatrix"][rd_col[i]],
                                    speed=0.15, signalDots='same', noiseDots='direction'))
    t = time.time()
    frame_lst = np.arange(start=1, stop=frames + 1, step=1)
    for frameN in range(frames):
        if frameN in frame_lst:
            for i, i_n in enumerate(rd_col):
                if i_n == 0:
                    rd_col[i] = 358
                if i_n == 1:
                    rd_col[i] = 359
                else:
                    rd_col[i] = rd_col[i] - 2
            for i, stim in enumerate(stims):
                stim.color = col_matrix["fullcolormatrix"][rd_col[i]]

        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        fix.draw()
        for stim in stims:
            stim.draw()
        win.flip()
    return t


def motion_selection(win, defaultKeyboard, c1, c2, c3, c4, d1, d2, d3, d4, correct, timer=4):
    frames = time_to_frame(timer, frameDur)
    stim1 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(-5.5, -5.5),
                                fieldShape='circle', dir=d1, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c1, name='DownLeft', speed=0.15, signalDots='different', noiseDots='direction')
    stim2 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(-5.5, 5.5),
                                fieldShape='circle', dir=d2, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c2, name='UpLeft', speed=0.15, signalDots='different', noiseDots='direction')
    stim3 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(5.5, 5.5),
                                fieldShape='circle', dir=d3, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c3, name='UpRight', speed=0.15, signalDots='different', noiseDots='direction')
    stim4 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(5.5, -5.5),
                                fieldShape='circle', dir=d4, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c4, name='DownRight', speed=0.15, signalDots='different', noiseDots='direction')

    circle_1 = visual.ShapeStim(win=win, vertices='circle', units='deg', size=8.1, pos=(-5.5, -5.5),
                                name='DownLeft', fillColor=None, lineColor=None, colorSpace='rgb255', interpolate=True)
    circle_2 = visual.ShapeStim(win=win, vertices='circle', units='deg', size=8.1, pos=(-5.5, 5.5),
                                name='UpLeft', fillColor=None, lineColor=None, colorSpace='rgb255', interpolate=True)
    circle_3 = visual.ShapeStim(win=win, vertices='circle', units='deg', size=8.1, pos=(5.5, 5.5),
                                name='UpRight', fillColor=None, lineColor=None, colorSpace='rgb255', interpolate=True)
    circle_4 = visual.ShapeStim(win=win, vertices='circle', units='deg', size=8.1, pos=(5.5, -5.5),
                                name='DownRight', fillColor=None, lineColor=None, colorSpace='rgb255', interpolate=True)
    fix = visual.ShapeStim(win=win, name='Fixation', vertices='circle', units='deg', size=(0.25, 0.25),
                           pos=(0, 0), ori=0.0,
                           fillColor='white', lineColor=None, colorSpace='rgb255', interpolate=True)
    resp_dot = visual.ImageStim(win=win, name='resp_dot',
                                image=f'{fn_stimuli}/mouse_cursor_black-modified.png', mask=None,
                                ori=0.0, pos=(0, 0), size=(0.75, 0.75), color=[1, 1, 1], colorSpace='rgb',
                                opacity=1, flipHoriz=False, flipVert=False,
                                texRes=128, interpolate=True, depth=-1.0)

    Clock = core.Clock()
    clicky = event.Mouse(win=win)
    speed = 12  # trackball speed multiplier

    clicky.mouseClock = core.Clock()
    clicky.setPos(newPos=(0, 0))
    clicky.setVisible(False)
    clicky.clicked_name = []

    trialComponents = [clicky, circle_1, circle_2, circle_3, circle_4, fix]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Clock.reset(-_timeToFirstFrame)
    response_onset = time.time()
    validClick = False
    t = time.time()

    for frameN in range(frames):
        resp_dot.setPos(clicky.getPos() * speed)
        pos = resp_dot.pos
        stim1.draw()
        stim2.draw()
        stim3.draw()
        stim4.draw()
        circle_1.draw()
        circle_2.draw()
        circle_3.draw()
        circle_4.draw()
        fix.draw()
        resp_dot.draw()

        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()

        if clicky.status == NOT_STARTED and t >= 0.0 - frameTolerance:
            clicky.frameNStart = frameN
            clicky.tStart = t
            win.timeOnFlip(clicky, 'tStartRefresh')
            clicky.status = STARTED
            clicky.mouseClock.reset()
            prevButtonState = clicky.getPressed()

        if clicky.status == STARTED:
            buttons = clicky.getPressed()
            if buttons != prevButtonState:
                prevButtonState = buttons
                if sum(buttons) > 0:
                    try:
                        iter([circle_1, circle_2, circle_3, circle_4])
                        clickableList = [circle_1, circle_2, circle_3, circle_4]
                    except:
                        clickableList = [[circle_1, circle_2, circle_3, circle_4]]
                    non_clickableList = [stim1, stim2, stim3, stim4]
                    while validClick == False:
                        for o, obj in enumerate(clickableList):
                            if obj.contains(pos):
                                clicky.clicked_name.append(obj.name)
                                obj.lineWidth = 2.5
                                if obj.name == correct:
                                    obj.lineColor = [0, 168, 0]
                                else:
                                    obj.lineColor = [186, 37, 37]
                                select_time = time.time()
                                trial_response = [np.array([non_clickableList[o].dir]),
                                                  np.array([non_clickableList[o].color]), obj.name, select_time]
                                clicky.clickReset()
                                clickableList.remove(obj)
                                validClick = True
                        if validClick == False:
                            clicky.clickReset()
                            break

                        if validClick == True:
                            for o, obj in enumerate(clickableList):
                                if obj.name == correct:
                                    obj.lineWidth = 2.5
                                    obj.lineColor = [0, 168, 0]
                            circle_1.draw()
                            circle_2.draw()
                            circle_3.draw()
                            circle_4.draw()

        win.flip()

    if validClick == False:
        trial_response = [np.array([999]), np.array([999, 999, 999]), "NoClick", 999]

    corr = 0
    if trial_response[2] == correct:
        corr = 2

    event.Mouse(visible=False)
    return response_onset, trial_response[0], trial_response[1], trial_response[2], trial_response[3], corr


def motion_sequence(con, startval_lst, fix_color, t0, win, defaultKeyboard, pre_time):
    ran = np.arange(1, 36, 1)
    mot_range = {"m1": 54, "m2": 125, "m3": 245}

    r = con["Round"]
    s = con["Set"]
    trial = con["Trial Condition"]

    c1 = colortriangle.loc[(colortriangle["color"] == con["Col1_n"]) & (colortriangle["idx"] == con["Col1_i"])]
    c1 = c1[["R", "G", "B"]].values.tolist()[0]
    c2 = colortriangle.loc[(colortriangle["color"] == con["Col2_n"]) & (colortriangle["idx"] == con["Col2_i"])]
    c2 = c2[["R", "G", "B"]].values.tolist()[0]

    m1 = mot_range[con["Dir1_n"]] + con["Dir1_i"]
    m2 = mot_range[con["Dir2_n"]] + con["Dir2_i"]

    rm1 = con["DLDir"]
    rc1 = con["DLCol"]
    rm2 = con["ULDir"]
    rc2 = con["ULCol"]
    rm3 = con["URDir"]
    rc3 = con["URCol"]
    rm4 = con["DRDir"]
    rc4 = con["DRCol"]
    cue = con["Cue"]
    stair = con["Staircase"]
    target = con["Target"]

    rm_lst = [rm1, rm2, rm3, rm4]
    rm_names = [rm1, rm2, rm3, rm4]
    rc_lst = [rc1, rc2, rc3, rc4]
    rc_names = [rc1, rc2, rc3, rc4]

    for i, i_n in enumerate(rm_lst):
        if i_n == con["Dir1_n"]:
            rm_lst[i] = m1
        if i_n == con["Dir2_n"]:
            rm_lst[i] = m2
        if i_n == f"{con['Dir1_n']}_a":
            sign = 1
            if con["Dir1_i"] + startval_lst[con["Dir1_n"]] > len(ran):
                sign = -1
            m_idx = con["Dir1_i"] + sign * startval_lst[con["Dir1_n"]]
            if m_idx <= 0:
                m_idx = 1
            rm_lst[i] = m1 + sign * m_idx
        if i_n == f"{con['Dir2_n']}_a":
            sign = 1
            if con["Dir2_i"] + startval_lst[con["Dir2_n"]] > len(ran):
                sign = -1
            m_idx = con["Dir2_i"] + sign * startval_lst[con["Dir2_n"]]
            if m_idx <= 0:
                m_idx = 1
            rm_lst[i] = m2 + sign * m_idx

    for i, i_n in enumerate(rc_lst):
        if i_n == con["Col1_n"]:
            rc_lst[i] = c1
        if i_n == con["Col2_n"]:
            rc_lst[i] = c2
        if i_n == f"{con['Col1_n']}_a":
            sign = 1
            if con['Col1_i'] + startval_lst[con['Col1_n']] >= len(ran):
                print(f'C_idx BEFORE: {con["Col1_i"] + sign * startval_lst[con["Col1_n"]]}')
                sign = -1
                print(sign)
                print(f"Index change to: {con['Col1_i'] - startval_lst[con['Col1_n']]}")
            c_idx = con["Col1_i"] + sign * startval_lst[con['Col1_n']]
            print(f"C index: {c_idx}")
            if c_idx <= 0:
                c_idx = 1
                print(f"Index change to: {con['Col1_i'] - startval_lst[con['Col1_n']]}")
            rc_a = colortriangle.loc[(colortriangle["color"] == con["Col1_n"]) & (
                    colortriangle["idx"] == c_idx)]
            print(f"rc_a: {rc_a}")
            rc_lst[i] = rc_a[["R", "G", "B"]].values.tolist()[0]

        if i_n == f"{con['Col2_n']}_a":
            sign = 1
            if con['Col2_i'] + startval_lst[con['Col2_n']] >= len(ran):
                print(f'C_idx BEFORE: {con["Col2_i"] + sign * startval_lst[con["Col2_n"]]}')
                sign = -1
                print(sign)
                print(f"Index change to: {con['Col2_i'] - startval_lst[con['Col2_n']]}")
            c_idx = con["Col2_i"] + sign * startval_lst[con['Col2_n']]
            print(f"C index: {c_idx}")
            if c_idx <= 0:
                c_idx = 1
                print(f"Index change to: {con['Col2_i'] - startval_lst[con['Col2_n']]}")
            rc_a = colortriangle.loc[(colortriangle["color"] == con["Col2_n"]) & (
                    colortriangle["idx"] == c_idx)]
            print(f"rc_a: {rc_a}")
            rc_lst[i] = rc_a[["R", "G", "B"]].values.tolist()[0]

    rm1, rm2, rm3, rm4 = rm_lst
    rc1, rc2, rc3, rc4 = rc_lst
    t_s1 = con["T_stim1"]
    t_s2 = con["T_stim2"]
    t_m1 = con["T_mask1"]
    t_m2 = con["T_mask2"]
    t_a = con["T_ans"]
    t_f = con["T_fix"]

    # Adapt ITI to compensate for timing drift
    post_time = time.time()
    trial_t = 16.8
    if con.name == 0:
        trial_t = 0
    diff = (post_time - pre_time) - trial_t

    iti = con["T_iti"] - diff - 0.001
    route_fix(win, defaultKeyboard, basic_col, timer=iti)
    print(f"ITI END Time: {time.time()}")

    # On jitter trials, wait for the next scanner trigger ("5") before proceeding
    if con["Jitter"] == 1:
        len5 = len(lst5["5"])
        while len5 == len(lst5["5"]):
            if check_for_keys("5"):
                lst5["5"].append(5)
                lst5["time"].append(time.time() - t0)
                print(f"Time of 5 pressed!: {time.time()}")

    pre_t = time.time()

    ts1 = motion_series1(win, defaultKeyboard, m1, c1, "normal", t_s1)
    tm1 = motion_mask(win, defaultKeyboard, t_m1)
    ts2 = motion_series1(win, defaultKeyboard, m2, c2, "normal", t_s2)
    tm2 = motion_mask(win, defaultKeyboard, t_m2)
    fix_onset = route_fix(win, defaultKeyboard, fix_color, timer=t_f)
    resp_onset, click_dir, click_col, click_pos, click_onset, correct = motion_selection(win, defaultKeyboard,
                                                                                          rc1, rc2, rc3, rc4,
                                                                                          rm1, rm2, rm3, rm4,
                                                                                          con["Correct"], t_a)
    trial_end = time.time()

    clicky = "error"
    if cue == "m":
        for m, m_n in enumerate(rm_lst):
            if m_n == click_dir:
                clicky = rm_names[m]
    elif cue == "c":
        for c, c_n in enumerate(rc_lst):
            if np.all(np.round(c_n, 0) == np.round(click_col[0], 0)):
                clicky = rc_names[c]
    if correct == 2:
        error = 0
    elif correct == 0 and clicky != "error":
        error = 2
        if clicky == f"{target}_a":
            error = 1
    else:
        error = 3

    trial = r, s, trial, con["ResStyle"], (t0), (pre_t - t0), (ts1 - t0), (tm1 - t0), (ts2 - t0), (tm2 - t0), \
            (fix_onset - t0), (resp_onset - t0), (trial_end - t0), \
            m1, c1[0], c1[1], c1[2], m2, c2[0], c2[1], c2[2], cue, rm1, rc1[0], rc1[1], rc1[2], rm2, rc2[0], \
            rc2[1], rc2[2], rm3, rc3[0], rc3[1], rc3[2], rm4, rc4[0], rc4[1], rc4[2], \
            (click_onset - t0), click_dir, click_pos, error, correct
    return trial, pre_t


# ====================== 3. Set up results & staircases ======================

basic_col = [226, 226, 226]
event.Mouse(visible=False)

globalClock = core.Clock()
routineTimer = core.CountdownTimer()

header = ["Round", "Set", "Trial", "ResStyle",
          "ExpTimeStart", "Trial_t", "Stim1_t", "Mask1_t", "Stim2_t", "Mask2_t", "Fix_t", "Response_t", "Trial_End",
          "Dir1", "Col1.1", "Col1.2", "Col1.3", "Dir2", "Col2.1", "Col2.2", "Col2.3",
          "Cue", "DLDir", "DLCol1", "DLCol2", "DLCol3", "ULDir", "ULCol1", "ULCol2", "ULCol3",
          "URDir", "URCol1", "URCol2", "URCol3", "DRDir", "DRCol1", "DRCol2", "DRCol3",
          "Click Onset", "Clicked", "ClickedPos", "Error", "Trial Correct"]

results_pd = pd.DataFrame([], columns=header)

round_start = int(expInfo["RoundStart"])
if expInfo["Run"] == 'single':
    print("Single Run")
    round_end = int(expInfo["RoundStart"])
else:
    print("Multiple Runs")
    round_end = 8

round_lst = np.arange(round_start, round_end + 1)

header_quest = ["Round", "Trial", "Target", "ErrorType",
                "QuestInt_c1", "QuestInt_c2", "QuestInt_c3", "QuestInt_m1", "QuestInt_m2", "QuestInt_m3",
                "QuestMean_c1", "QuestMean_c2", "QuestMean_c3", "QuestMean_m1", "QuestMean_m2", "QuestMean_m3"]
results_quest = pd.DataFrame([[0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0]], columns=header_quest)

# Staircases
c1 = data.QuestHandler(name="c1", startVal=8, startValSd=9, method='quantile', range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None, minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)
c2 = data.QuestHandler(name="c2", startVal=11, startValSd=9, method='quantile', range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None, minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)
c3 = data.QuestHandler(name="c3", startVal=11, startValSd=9, method='quantile', range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None, minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)
m1 = data.QuestHandler(name="m1", startVal=9, startValSd=9, method='quantile', range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None, minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)
m2 = data.QuestHandler(name="m2", startVal=12, startValSd=9, method='quantile', range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None, minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)
m3 = data.QuestHandler(name="m3", startVal=7, startValSd=9, method='quantile', range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None, minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)

questopt = {"c1": c1, "c2": c2, "c3": c3, "m1": m1, "m2": m2, "m3": m3}

# Load staircase from previous sessions if available
for q, q_n in questopt.items():
    print(q_n.name)
    if os.path.exists(f"{fn_quest}/quest_{str(q_n.name)}.psydat"):
        print(f"Loaded {q_n.name} staircase from past sessions")
        questopt[q] = fromFile(f"{fn_quest}/quest_{str(q_n.name)}.psydat")
        print(questopt[q]._nextIntensity)
        print(questopt[q].intensities)
    else:
        print(f" {fn_quest}/quest_{str(q_n.name)}.psydat does not exist")


# ====================== 4. Start the experiment ======================

event.Mouse(visible=False)

t0 = time.time()
lst5 = {"5": [], "time": []}

text = "Aufgabe 1: Binding Experiment"
route_txt_cent(text, basic_col, win, defaultKeyboard, 5)
print(round_lst)

for run, i_r in enumerate(round_lst):
    print(f"Round {run}")

    # Wait for experimenter to press Return before starting each round
    enter_lst = []
    while len(enter_lst) == 0:
        if check_for_keys("return"):
            enter_lst.append("return")

    trial_con = conditions[conditions["Round"] == i_r][:]
    trial_con.reset_index(inplace=True, drop=True)
    trial_lst = np.arange(1, len(trial_con) + 1)

    txt_round = f"Runde {i_r}"
    route_txt_cent(txt_round, basic_col, win, defaultKeyboard, 1)

    # Wait for 5 scanner triggers before starting the block
    len5 = len(lst5["5"])
    while len(lst5["5"]) < len5 + 5:
        if check_for_keys("5"):
            lst5["5"].append(5)
            lst5["time"].append(time.time() - t0)

    pre_time = time.time()
    for i, i_n in enumerate(trial_lst):
        startval_lst = {"m1": questopt["m1"]._nextIntensity, "m2": questopt["m2"]._nextIntensity,
                        "m3": questopt["m3"]._nextIntensity, "c1": questopt["c1"]._nextIntensity,
                        "c2": questopt["c2"]._nextIntensity, "c3": questopt["c3"]._nextIntensity}

        con = trial_con.iloc[i]
        results, pre_time = motion_sequence(con, startval_lst, basic_col, t0, win, defaultKeyboard, pre_time)
        results = pd.DataFrame([results], columns=header)

        results_pd = pd.concat([results_pd, results], ignore_index=True, axis=0)
        [os.remove(fn) for fn in [fn_excel, fn_hdf5, fn_excel_lst5, fn_hdf5_lst5] if os.path.exists(fn)]
        results_pd.to_excel(fn_excel)
        results_pd.to_hdf(fn_hdf5, key="bindingexp_mri", mode="w")
        lst5_df = pd.DataFrame(lst5)
        lst5_df.to_excel(fn_excel_lst5)
        lst5_df.to_hdf(fn_hdf5_lst5, key="tr_mri", mode="w")

    # Block feedback and end-of-block wait (until scanner stops sending triggers)
    feedback(results_pd, i_r)
    route_fix(win, defaultKeyboard, basic_col, timer=10)
    end_exp = False
    time_left = time.time()
    lst_new = []
    while end_exp == False:
        while time.time() - time_left < 2:
            if check_for_keys("5"):
                lst_new.append(1)
        if len(lst_new) == 0:
            end_exp = True
            break
        else:
            time_left = time.time()
            lst_new = []

    results_round = results_pd[results_pd["Round"] == i_r][:]
    txt_round_end = f"Runde {i_r} ist geschafft! Bleib ruhig liegen, gleich geht es weiter."
    route_txt_cent(txt_round_end, basic_col, win, defaultKeyboard, 55)


# ====================== 5. End session ======================

txt_end = f'Gut gemacht! \n \nDas war der letzte Block. \n \nBitte bleiben Sie still liegen.'
route_txt_cent(txt_end, basic_col, win, defaultKeyboard, 60)

# Wait until scanner stops sending triggers
end_exp = False
time_left = time.time()
lst_new = []
while end_exp == False:
    while time.time() - time_left < 2:
        if check_for_keys("5"):
            lst_new.append(1)
    if len(lst_new) == 0:
        end_exp = True
        break
    else:
        time_left = time.time()
        lst_new = []

win.flip()
win.close()
core.quit()