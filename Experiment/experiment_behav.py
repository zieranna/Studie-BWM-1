from statistics import mean
import os
import math
import time
import shutil
from random import sample
import matplotlib.pyplot as plt
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
# Monitor parameters — adapt to your setup
# gammaGrid: [[min, max, gamma, "nan", "nan", "nan"], ...] per channel (all, R, G, B)
gammaGrid = ___  # e.g. [[0.42, 346.58, 2.12, "nan", "nan", "nan"], ...]
mon = monitors.Monitor('___')  # monitor name as registered in PsychoPy Monitor Center
mon.setDistance(___)           # viewing distance in cm
mon.setWidth(___)              # screen width in cm
mon.setSizePix([___, ___])     # screen resolution [width, height]
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
expInfo = {'Participant': '04', 'Session N°': '1', 'RoundStart': '1', 'Run': 'all'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()
expInfo['date'] = data.getDateStr()
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

win.fullscr = True

# ================================================================================
# Directory structure:
#   <fn_base>/
#     sub-<ID>/
#       Input/
#         beh/
#           conditions_<session>.xlsx     ← trial conditions per session
#         stimuli/                        ← ColorWheel360.mat, Look_up_table.xlsx, mouse cursor image
#       Output/
#         beh/
#           BindingExp/
#             <session>/                  ← trial results (xlsx, hdf5), staircase plots
#               click/                    ← per-trial cursor trajectory CSVs
#         behQuest/                       ← staircase .psydat and .xlsx files (persist across sessions)

fn_base = "___"  # base directory, e.g. "C:/Users/.../participants"
fn_stimuli = "___"  # stimuli directory containing ColorWheel360.mat, Look_up_table.xlsx, mouse cursor image

fn_subject = f"{fn_base}/sub-{expInfo['Participant']}"
fn_results = f"{fn_subject}/Output/beh/BindingExp/{expInfo['Session N°']}"
fn_results_click = f"{fn_subject}/Output/beh/BindingExp/{expInfo['Session N°']}/click"
fn_quest = f"{fn_subject}/Output/behQuest"
[os.makedirs(fn) if not os.path.exists(fn) else print(f" {fn} Folder already exists") for fn in (fn_results, fn_quest, fn_results_click)]

fn_condition = f"{fn_subject}/Input/beh/conditions_{expInfo['Session N°']}.xlsx"
conditions = pd.read_excel(fn_condition)
shutil.copy(fn_condition, f"{fn_results}/Input_conditions.xlsx")

# Output file names
fn_hdf5 = fn_results + '/binding_exp_results.hdf5'
fn_excel = f"{fn_results}/binding_exp_results.xlsx"
fn_excel_quest = f"{fn_results}/totalquest_results.xlsx"
fn_hdf5_quest = f"{fn_results}/totalquest_results.hdf5"

fn_excel_bac = f"{fn_results}/binding_exp_bac_{expInfo['date']}.xlsx"
fn_hdf5_bac = f"{fn_results}/binding_exp_bac_{expInfo['date']}.hdf5"
fn_quest_bac = f"{fn_results}/totalquest_bac_{expInfo['date']}.xlsx"
fn_quest_hdf5_bac = f"{fn_results}/totalquest_bac_{expInfo['date']}.hdf5"

[os.rename(fn, fn_bac) for fn, fn_bac in zip([fn_excel, fn_hdf5, fn_excel_quest, fn_hdf5_quest], [fn_excel_bac, fn_hdf5_bac, fn_quest_bac, fn_quest_hdf5_bac]) if os.path.exists(fn)]

endExpNow = False
frameTolerance = 0.001

expInfo['frameRate'] = win.getActualFrameRate()
print(f"Frame Rate: {expInfo['frameRate']}")
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.256

ioSession = ioServer = eyetracker = None

defaultKeyboard = keyboard.Keyboard()
keys = keyboard.Keyboard()

if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
    core.quit()

# ================================================================================
# Load stimulus look-up tables
colortriangle = pd.read_excel(f"{fn_stimuli}/Look_up_table.xlsx")
colortriangle = colortriangle[["color", "idx", "R_a42", "G_a42", "B_a42"]]
colortriangle = colortriangle.rename(columns={"R_a42": "R", "G_a42": "G", "B_a42": "B"})
print(colortriangle)
colortriangle_rgb = colortriangle[["R", "G", "B"]]
col_matrix = scipy.io.loadmat(f'{fn_stimuli}/ColorWheel360.mat')


# ====================== 2. Set up the functions for sequences ======================

def route_txt_long(txt, col, box_size, key_resp, win, defaultKeyboard):
    """
    Displays a long text in a textbox (i.e. for instructions) and allows the participant to
    continue when he presses Enter or Space. Text is left-aligned.
    """
    import textwrap
    text_sentences = textwrap.wrap(txt, width=box_size)
    text_blocks = [text_sentences[i:i + int(box_size/2)] for i in range(0, len(text_sentences), int(box_size/2))]
    text_blocks = [' '.join(text_blocks[i]) for i in range(0, len(text_blocks))]
    for i in range(len(text_blocks)):
        text_blocks[i] = str(text_blocks[i]).replace("', '", '').replace('[', '').replace(']', '')

    for block in text_blocks:
        text = visual.TextBox2(win=win, text=block, size=(box_size, box_size/2), letterHeight=1,
                               color=col, colorSpace='rgb255', pos=(0, 4), units='deg', alignment='left')
        text_enter = visual.TextBox2(win=win, text='(Drücke Enter um fortzufahren)', size=(10, 4), letterHeight=1,
                                     color=col, colorSpace='rgb255', pos=(0, -8), units='deg', alignment='center')
        Clock = core.Clock()
        continueRoutine = True
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        welcomeComponents = [key_resp]
        for thisComponent in welcomeComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        Clock.reset(-_timeToFirstFrame)
        frameN = -1

        while continueRoutine:
            text.draw()
            text_enter.draw()
            win.flip()

            if key_resp.status == NOT_STARTED:
                key_resp.frameNStart = frameN
                key_resp.tStart = t
                win.timeOnFlip(key_resp, 'tStartRefresh')
                key_resp.status = STARTED
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')
            if key_resp.status == STARTED:
                theseKeys = key_resp.getKeys(keyList=['space', 'return'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    continueRoutine = False

            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()

            if not continueRoutine:
                break

            for thisComponent in welcomeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break

        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)


def route_txt_cent(txt, col, key_resp, win, defaultKeyboard):
    """
    Displays a short centered text (e.g. round announcement) and waits for Enter or Space.
    """
    text = visual.TextBox2(win=win, text=txt, size=(10, 4), letterHeight=1,
                           color=col, colorSpace='rgb255', pos=(0, 4), units='deg', bold=True, alignment='center')
    text_enter = visual.TextBox2(win=win, text='(Drücke Enter um fortzufahren)', size=(10, 4), letterHeight=1,
                                 color=col, colorSpace='rgb255', pos=(0, -8), units='deg', alignment='center')
    Clock = core.Clock()
    continueRoutine = True
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    welcomeComponents = [key_resp]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Clock.reset(-_timeToFirstFrame)
    frameN = -1

    while continueRoutine:
        text.draw()
        text_enter.draw()
        win.flip()

        if key_resp.status == NOT_STARTED:
            key_resp.frameNStart = frameN
            key_resp.tStart = t
            win.timeOnFlip(key_resp, 'tStartRefresh')
            key_resp.status = STARTED
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')
        if key_resp.status == STARTED:
            theseKeys = key_resp.getKeys(keyList=['space', 'return'], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name
                key_resp.rt = _key_resp_allKeys[-1].rt
                continueRoutine = False

        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()

        if not continueRoutine:
            break
        continueRoutine = False
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break

    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)


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


def route_fix(win, defaultKeyboard, col, timer=None, dot_size=(0.25, 0.25)):
    """
    Creates a fixation point.

    If timer is None: returns the fixation stimulus for use in another routine.
    If timer is given: displays fixation for that duration and returns the onset time.
    """
    fix = visual.Circle(win=win, name='Fixation', units='deg', size=dot_size, pos=(0, 0), ori=0.0,
                        fillColor=col, lineColor=None, colorSpace='rgb255', interpolate=True)

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
                                color=col1,
                                speed=0.25, signalDots='different', noiseDots='direction')
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
    frame_lst = np.arange(start=1, stop=frames+1, step=1)
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


def motion_selection(win, defaultKeyboard, c1, c2, c3, c4, d1, d2, d3, d4, correct, trial, timer=4):
    frames = time_to_frame(timer, frameDur)
    stim1 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(-5.5, -5.5),
                                fieldShape='circle', dir=d1, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c1, name='DownLeft',
                                speed=0.15, signalDots='different', noiseDots='direction')
    stim2 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(-5.5, 5.5),
                                fieldShape='circle', dir=d2, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c2, name='UpLeft',
                                speed=0.15, signalDots='different', noiseDots='direction')
    stim3 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(5.5, 5.5),
                                fieldShape='circle', dir=d3, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c3, name='UpRight',
                                speed=0.15, signalDots='different', noiseDots='direction')
    stim4 = deadDots.zombieDots(win=win, fieldSize=8, units="deg", nDots=150, dotLife=-1, fieldPos=(5.5, -5.5),
                                fieldShape='circle', dir=d4, coherence=1, dotSize=7, colorSpace='rgb255',
                                color=c4, name='DownRight',
                                speed=0.15, signalDots='different', noiseDots='direction')

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
    speed = 1

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
    frameN = -1
    response_onset = time.time()
    validClick = False
    t = time.time()
    click_pos_eval = []

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

        if frameN % 4 == 0:
            click_data = pos[0], pos[1], frameN
            click_pos_eval.append(click_data)

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

    click_pos_eval = pd.DataFrame(click_pos_eval, columns=["x", "y", "frame"])
    click_pos_eval.to_csv(f'{fn_results_click}/clicked_pos{trial}.csv')

    event.Mouse(visible=False)
    return response_onset, trial_response[0], trial_response[1], trial_response[2], trial_response[3], corr


def motion_sequence(con, startval_lst, fix_color, t0, win, defaultKeyboard, pre_time):
    colorwheel = scipy.io.loadmat(f'{fn_stimuli}/ColorWheel360.mat')
    colorwheel = colorwheel['fullcolormatrix']

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
            if con["Dir1_i"] + startval_lst[con["Dir1_n"]] >= len(ran):
                sign = -1
            rm_lst[i] = m1 + sign * startval_lst[con['Dir1_n']]
        if i_n == f"{con['Dir2_n']}_a":
            sign = 1
            if con["Dir2_i"] + startval_lst[con["Dir2_n"]] >= len(ran):
                sign = -1
            rm_lst[i] = m2 + sign * startval_lst[con['Dir2_n']]

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
                startval_lst[con['Col1_n']] == con['Col1_i'] - 1
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
                startval_lst[con['Col2_n']] == con['Col2_i'] - 1
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
    print(f"trial duration: {post_time - pre_time}")
    print(f"Time difference: {diff}")

    iti = con["T_iti"] - diff
    if iti <= 0:
        iti = 1.2
    print(f"ITI: {iti}")
    if con["Jitter"] == 1:
        iti = iti - 0.1
    route_fix(win, defaultKeyboard, fix_color, timer=iti)

    pre_t = time.time()

    pre_st = time.time()
    ts1 = motion_series1(win, defaultKeyboard, m1, c1, "normal", t_s1)
    print(f"Duration Stim1: {time.time()-pre_st}")
    pre_st = time.time()
    tm1 = motion_mask(win, defaultKeyboard, t_m1)
    print(f"Duration Mask1: {time.time()-pre_st}")
    pre_st = time.time()
    ts2 = motion_series1(win, defaultKeyboard, m2, c2, "normal", t_s2)
    print(f"Duration Stim2: {time.time()-pre_st}")
    pre_st = time.time()
    tm2 = motion_mask(win, defaultKeyboard, t_m2)
    print(f"Duration Mask2: {time.time()-pre_st}")
    pre_st = time.time()
    fix_onset = route_fix(win, defaultKeyboard, fix_color, timer=t_f)
    print(f"Duration fixation: {time.time()-pre_st}")
    pre_st = time.time()
    resp_onset, click_dir, click_col, click_pos, click_onset, correct = motion_selection(win, defaultKeyboard,
                                                                                          rc1, rc2, rc3, rc4,
                                                                                          rm1, rm2, rm3, rm4,
                                                                                          con["Correct"], trial, t_a)
    print(f"Duration Response: {time.time()-pre_st}")

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

    trial = r, s, trial, con["ResStyle"], (t0), (pre_t - t0), (ts1 - t0), (tm1 - t0), (ts2 - t0), (tm2 - t0), (fix_onset - t0), (resp_onset - t0), \
            m1, c1[0], c1[1], c1[2], m2, c2[0], c2[1], c2[2], cue, rm1, rc1[0], rc1[1], rc1[2], rm2, rc2[0], \
            rc2[1], rc2[2], rm3, rc3[0], rc3[1], rc3[2], rm4, rc4[0], rc4[1], rc4[2], \
            (click_onset - t0), click_dir, click_pos, error, correct

    return trial, pre_t


# ====================== 3. Display instructions & prepare stimuli ======================

txt_welcome = f'Hallo, \n \nWillkommen bei meinem Experiment! \nIch bedanke mich herzlich bei Ihnen für Ihre Teilnahme.\n \nLiebe Grüße, Anna'
txt_instructions = 'Dieses Experiment testet ihr visuelles Arbeitsgedächtnis. Es besteht aus einer Serie von Durchgängen. Ein Durchgang (ca. 16 Sek) besteht aus einer Abfolge von Ereignissen, die nun kurz beschrieben werden. In jedem Durchgang werden Sie aufgefordert, sich zwei Reize zu merken. Bei den Reizen handelt es sich um farbige Punktewolken, die in eine bestimmte Richtung fliegen. \n \nDiese Punktwolken werden aufeinanderfolgend abgespielt. Direkt nach jede Punktewolke wird zusätzlich noch eine „Maskierungswolke" kurz präsentiert. Diese hat keine bestimmte Richtung und Farbe, ist nicht relevant für die Gedächtnisaufgabe und dient lediglich dazu mögliche Nachbilder zu verhindern. Folgend ein Beispiel. Die zu merkenden Punktewolken mit ihrer jeweils spezifischen Farb-Richtungs-Kombination sollen Sie sich möglichst visuell vorstellen und so im Gedächtnis über eine Zeitspanne von ca. 10 Sek. halten. Die Abfrage, wie gut sie sich diese Punktewolken merken können, erfolgt über einen Testbildschirm, bei dem 4 neue Punktewolken gleichzeitig präsentiert werden.  Ihre Aufgabe ist nun unter diesen vier neuen Punktewolken, die erst genannte zu identifizieren und sie anzuklicken. Nur eine der 4 neunen Punktewolken ist einer der zwei zu merkenden Punktewolken sehr ähnlich, das heißt, sie hat die gleiche Richtung, unterscheidet sich aber ganz leicht in der Farbe oder aber sie hat die gleiche Farbe, unterscheidet sich aber leicht in der Richtung. Die anderen drei Testwolken, weichen leicht in beiden Merkmalen ab (das heißt, die Farbe und die Orientierung ist ähnlich einer der gemerkten Punktewolke) oder aber sie wurden falsch zusammengesetzt (z.B. die Richtung der einen zu merkenden Wolke wurde mit der leicht anderer Farbe der anderen zu merkenden Wolke kombiniert oder andersrum). Sie sollen also die Punktewolke auswählen, die in einem Merkmal identisch und in dem anderen leicht verschieden zu einer der zwei gemerkten Wolken ist. Wichtig: \n \nIn der Mitte des Bildschirms erscheint über die Dauer des gesamten Experiment eine Fixationshilfe. Fixieren Sie bitte mit den Augen immer die Mitte des Bildschirms.\n \nBitte stellen Sie Fragen, falls noch Unklarheiten bestehen oder Sie sich nicht sicher sind, ob Sie die Aufgabe richtig durchführen.  Die Auswahl kann nicht mehr korrigiert werden. Wählen Sie deshalb ihre Antwort vorsichtig aus. Sie haben Sie insgesamt 5 Sekunden Zeit dafür. Die Durchgänge werden durch eine Pause von 3-8 Sek. voneinander getrennt. 18 Durchgänge werden zu einem Aufgabenblock zusammengefasst, der etwa 7 Minuten dauert. Insgesamt werden acht Blöcke der Aufgabe präsentiert. Zwischen den Blöcken können Sie eine kurze Pause machen. Gibt es noch Fragen? Jetzt geht es los!'

basic_col = [226, 226, 226]
event.Mouse(visible=False)

globalClock = core.Clock()
routineTimer = core.CountdownTimer()

route_txt_long(txt_welcome, basic_col, 40, keys, win, defaultKeyboard)
route_txt_long(txt_instructions, basic_col, 40, keys, win, defaultKeyboard)

header = ["Round", "Set", "Trial", "ResStyle",
          "ExpTimeStart", "Trial_t", "Stim1_t", "Mask1_t", "Stim2_t", "Mask2_t", "Fix_t", "Response_t",
          "Dir1", "Col1.1", "Col1.2", "Col1.3", "Dir2", "Col2.1", "Col2.2", "Col2.3",
          "Cue", "DLDir", "DLCol1", "DLCol2", "DLCol3", "ULDir", "ULCol1", "ULCol2", "ULCol3",
          "URDir", "URCol1", "URCol2", "URCol3", "DRDir", "DRCol1", "DRCol2", "DRCol3",
          "Click Onset", "Clicked", "ClickedPos", "Error", "Trial Correct"]

results_pd = pd.DataFrame([], columns=header)

header_quest = ["Round", "Trial", "Target", "ErrorType",
                "QuestInt_c1", "QuestInt_c2", "QuestInt_c3", "QuestInt_m1", "QuestInt_m2", "QuestInt_m3",
                "QuestMean_c1", "QuestMean_c2", "QuestMean_c3", "QuestMean_m1", "QuestMean_m2", "QuestMean_m3"]
results_quest = pd.DataFrame([[0, 0, 0, 0, 8, 11, 11, 9, 12, 7, 0, 0, 0, 0, 0, 0]], columns=header_quest)

col_names = ["c1", "c2", "c3"]
col_startVal = [8, 11, 11]
mot_names = ["m1", "m2", "m3"]
mot_startVal = [9, 12, 7]
questopt = {"c1": None, "c2": None, "c3": None, "m1": None, "m2": None, "m3": None}
for col_ii, col in enumerate(col_names):
    questopt[col] = data.QuestHandler(name=col, startVal=col_startVal[col_ii], startValSd=9, method='quantile',
                                      range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None,
                                      minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)
for mot_ii, mot in enumerate(mot_names):
    questopt[mot] = data.QuestHandler(name=mot, startVal=mot_startVal[mot_ii], startValSd=9, method='quantile',
                                      range=20, beta=3.5, gamma=0.5, delta=0.05, staircase=None,
                                      minVal=1, maxVal=20, pThreshold=0.75, nTrials=1000)

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

round_start = int(expInfo["RoundStart"])
if expInfo["Run"] == 'single':
    print("Single Run")
    round_end = int(expInfo["RoundStart"])
else:
    print("Multiple Runs")
    round_end = 8

round_lst = np.arange(round_start, round_end + 1)

t0 = time.time()
for run, i_r in enumerate(round_lst):
    trial_con = conditions[conditions["Round"] == i_r][:]
    trial_con.reset_index(inplace=True, drop=True)
    trial_lst = np.arange(1, len(trial_con) + 1)
    print(f"Trial List: {trial_lst}")

    txt_round = f"Runde {i_r}"
    route_txt_cent(txt_round, basic_col, keys, win, defaultKeyboard)

    pre_trial = time.time()
    for i, i_n in enumerate(trial_lst):
        startval_lst = {"m1": round(questopt["m1"]._nextIntensity), "m2": round(questopt["m2"]._nextIntensity),
                        "m3": round(questopt["m3"]._nextIntensity), "c1": round(questopt["c1"]._nextIntensity),
                        "c2": round(questopt["c2"]._nextIntensity), "c3": round(questopt["c3"]._nextIntensity)}
        print(startval_lst)
        con = trial_con.iloc[i]

        results, pre_trial = motion_sequence(con, startval_lst, basic_col, t0, win, defaultKeyboard, pre_trial)
        results = pd.DataFrame([results], columns=header)
        print(startval_lst)

        if i_r > 1:
            print("After first round, update staircase")
            for q, q_n in enumerate(questopt):
                if f"{con['Staircase']}" == q_n:
                    print(f"Update staircase {q_n}")
                    if results["Error"][0] == 0:
                        if questopt[q_n].thisTrialN <= 0:
                            questopt[q_n].addResponse(1)
                        else:
                            questopt[q_n].addResponse(1, intensity=startval_lst[q_n])
                        questopt[q_n].next()
                    elif results["Error"][0] == 1:
                        print(f"Update staircase {q_n}")
                        if questopt[q_n].thisTrialN <= 0:
                            questopt[q_n].addResponse(0)
                        else:
                            questopt[q_n].addResponse(0, intensity=startval_lst[q_n])
                        questopt[q_n].next()
                    else:
                        pass

        mean_lst = [mean(questopt[q_n].data) if questopt[q_n].data != [] else 0 for q, q_n in enumerate(questopt)]

        results_pd = pd.concat([results_pd, results], ignore_index=True, axis=0)
        trial_quest = pd.DataFrame([[i_r, i_n, con["Staircase"], results["Error"][0],
                                     questopt["c1"]._nextIntensity, questopt["c2"]._nextIntensity,
                                     questopt["c3"]._nextIntensity, questopt["m1"]._nextIntensity,
                                     questopt["m2"]._nextIntensity, questopt["m3"]._nextIntensity,
                                     mean_lst[0], mean_lst[1], mean_lst[2], mean_lst[3], mean_lst[4], mean_lst[5]]],
                                    columns=header_quest)
        results_quest = pd.concat([results_quest, trial_quest], ignore_index=True, axis=0)

        [os.remove(fn) for fn in [fn_excel, fn_hdf5, fn_excel_quest, fn_hdf5_quest] if os.path.exists(fn)]
        results_pd.to_excel(fn_excel)
        results_pd.to_hdf(fn_hdf5, key="df", mode="w")
        results_quest.to_excel(fn_excel_quest)
        results_quest.to_hdf(fn_hdf5_quest, key="df", mode="w")

        [os.remove(f"{fn_quest}/quest_{q_n}.xlsx") for q, q_n in enumerate(questopt) if os.path.exists(f"{fn_quest}/quest_{q_n}.xlsx")]
        [questopt[q_n].saveAsExcel(f"{fn_quest}/quest_{q_n}", fileCollisionMethod='rename') for q, q_n in enumerate(questopt) if questopt[q_n].data != []]
        [os.remove(f"{fn_quest}/quest_{q_n}.psydat") for q, q_n in enumerate(questopt) if os.path.exists(f"{fn_quest}/quest_{q_n}.psydat")]
        [questopt[q_n].saveAsPickle(f"{fn_quest}/quest_{q_n}", fileCollisionMethod='rename') for q, q_n in enumerate(questopt) if questopt[q_n].data != []]

    results_round = results_pd[results_pd["Round"] == i_r][:]


# ====================== 5. End session ======================

txt_end = f'Gut gemacht! \n \nDu hast es geschafft! \n Vielen Dank. Bitte hole den Experimentleiter, um die nächste Aufgabe starten.'
route_txt_long(txt_end, basic_col, 40, keys, win, defaultKeyboard)

col_names = ["c1", "c2", "c3"]
mot_names = ["m1", "m2", "m3"]
all_quests = {"c1": [], "c2": [], "c3": [], "m1": [], "m2": [], "m3": []}
all_names = col_names + mot_names
for cond_name in all_names:
    all_quests[f"{cond_name}"] = (results_quest[results_quest["Target"] == cond_name][f"QuestInt_{cond_name}"]).reset_index(drop=True)

print(all_quests)
plt.figure()
for con, cond_name in enumerate(col_names):
    print(all_quests[f"{cond_name}"])
    plt.plot(np.arange(0, len(all_quests[f"{cond_name}"])), all_quests[f"{cond_name}"], label=cond_name)
plt.legend()
plt.savefig(f"{fn_results}/staircase_color.png")

plt.figure()
for con, cond_name in enumerate(mot_names):
    plt.plot(np.arange(0, len(all_quests[f"{cond_name}"])), all_quests[f"{cond_name}"], label=cond_name)
plt.legend()
plt.savefig(f"{fn_results}/staircase_motion.png")

win.flip()
win.close()
core.quit()