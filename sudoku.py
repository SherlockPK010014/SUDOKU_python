# -*- coding: utf-8 -*-
# แนะนำ Python 3.10/3.11
# pip install numpy==1.26.4 opencv-python==4.10.0.84 mediapipe==0.10.21 protobuf==4.25.3 cvzone==1.6.1

import cv2
import numpy as np
import random
import time
from cvzone.HandTrackingModule import HandDetector

# ===== UI / Control Config =====
GRID = 9
CELL = 56                 # ขนาดช่องกระดาน
BOARD_SIZE = GRID * CELL  # = 504 ถ้า CELL=56
LEFT = 20                 # ระยะจากขอบซ้าย
TOP = 40                  # ระยะจากขอบบน

MIRROR = True             # True = กล้องเซลฟี่ (flip ภาพ)
SWAP_HANDEDNESS_IF_MIRROR = True  # ถ้าเปิด จะสลับ Left<->Right เมื่อ MIRROR=True (แก้ label มือสลับ)
PINCH_THRESH = 45         # ใช้เฉพาะในบางจุด (ยังเก็บไว้เผื่อ)
OVERLAY_ALPHA = 0.25      # โปร่งใสของเลเยอร์ไฮไลต์/พื้นหลัง

# ท่าทาง/เวลารอ
GESTURE_HOLD = 0.35       # เวลาค้างท่าทาง (s) เพื่อคอมมิตตัวเลข
GESTURE_COOLDOWN = 0.6    # กันสั่งซ้ำ (s)
STRICT_VALIDATE = True    # True = ไม่ยอมวางเลขที่ผิดกติกา (ผิดกติกา = กะพริบกรอบแดงเตือน)

# ===== Sudoku helpers =====
def is_safe(grid, r, c, v):
    if v == 0:
        return True
    for j in range(9):
        if j != c and grid[r, j] == v:
            return False
    for i in range(9):
        if i != r and grid[i, c] == v:
            return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if (i, j) != (r, c) and grid[i, j] == v:
                return False
    return True

def find_empty(grid):
    for i in range(9):
        for j in range(9):
            if grid[i, j] == 0:
                return i, j
    return None

def solve(grid):
    empty = find_empty(grid)
    if not empty:
        return True
    r, c = empty
    nums = list(range(1, 10))
    random.shuffle(nums)
    for v in nums:
        if is_safe(grid, r, c, v):
            grid[r, c] = v
            if solve(grid):
                return True
            grid[r, c] = 0
    return False

def count_solutions(grid, limit=2):
    g = grid.copy()
    count = [0]
    def backtrack():
        if count[0] >= limit:
            return
        empty = find_empty(g)
        if not empty:
            count[0] += 1
            return
        r, c = empty
        for v in range(1, 10):
            if is_safe(g, r, c, v):
                g[r, c] = v
                backtrack()
                g[r, c] = 0
                if count[0] >= limit:
                    return
    backtrack()
    return count[0]

def generate_full():
    grid = np.zeros((9, 9), dtype=int)
    solve(grid)
    return grid

def generate_puzzle(target_empties=50):
    full = generate_full()
    puzzle = full.copy()
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    removed = 0
    for r, c in cells:
        if removed >= target_empties:
            break
        temp = puzzle[r, c]
        puzzle[r, c] = 0
        if count_solutions(puzzle, limit=2) != 1:
            puzzle[r, c] = temp
        else:
            removed += 1
    return puzzle, full

def new_game(target_empties=50):
    puzzle, full = generate_puzzle(target_empties)
    board = puzzle.copy()
    fixed = (puzzle != 0)
    return puzzle, full, board, fixed

# ===== Gesture mapping (มือขวา: นิ้วโป้ง-ก้อย) -> ตัวเลข =====
# fingersUp(): [thumb, index, middle, ring, pinky] เป็น 0/1
GESTURE_MAP_DIGIT = {
    (0,1,0,0,0): 1,
    (0,1,1,0,0): 2,
    (0,1,1,1,0): 3,
    (0,1,1,1,1): 4,
    (1,1,1,1,1): 5,  # กางมือทุกนิ้ว
    (1,0,0,0,0): 6,  # โป้งเดี่ยว
    (1,1,0,0,0): 7,  # โป้ง+ชี้
    (1,1,1,0,0): 8,  # โป้ง+ชี้+กลาง
    (1,1,1,1,0): 9,  # โป้ง+ชี้+กลาง+นาง
}

def classify_digit_right(fingers):
    tpl = tuple(int(x) for x in fingers)
    if tpl in GESTURE_MAP_DIGIT:
        return GESTURE_MAP_DIGIT[tpl]
    # fallback: ถ้าไม่ตรง mapping ชัดเจน ให้รองรับนับนิ้วไม่รวมโป้ง 1..4
    if fingers[0] == 0:
        c = sum(fingers[1:])
        if 1 <= c <= 4:
            return c
    return None

# ===== วาดทั้งหมดลงเฟรม =====
def draw_overlay(frame, board, selected, fixed_mask, invalid_until=None):
    overlay = frame.copy()
    ox, oy = int(LEFT), int(TOP)

    # ไฮไลต์ช่องที่เลือก (โปร่งใส)
    r, c = selected
    if 0 <= r < 9 and 0 <= c < 9:
        x1 = ox + c * CELL
        y1 = oy + r * CELL
        x2 = x1 + CELL
        y2 = y1 + CELL
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), -1)

    # กริดโปร่งใสชั้นแรก (ครบ 9×9)
    for i in range(10):
        thick = 1 if i % 3 else 2
        # เส้นแนวนอน
        cv2.line(overlay, (ox, oy + i * CELL), (ox + BOARD_SIZE, oy + i * CELL), (0, 255, 0), thick)
        # เส้นแนวตั้ง
        cv2.line(overlay, (ox + i * CELL, oy), (ox + i * CELL, oy + BOARD_SIZE), (0, 255, 0), thick)

    # ช่องผิดกติกา (จุดแดง)
    for i in range(9):
        for j in range(9):
            v = int(board[i, j])
            if v != 0 and not is_safe(board, i, j, v):
                cx = ox + j * CELL + CELL // 2
                cy = oy + i * CELL + CELL // 2
                cv2.circle(overlay, (int(cx), int(cy)), 6, (0, 0, 255), -1)

    # blend
    out = frame.copy()
    cv2.addWeighted(overlay, OVERLAY_ALPHA, out, 1 - OVERLAY_ALPHA, 0, out)

    # วาดเส้นกริดทับให้คมอีกครั้ง
    for i in range(10):
        thick = 1 if i % 3 else 2
        cv2.line(out, (ox, oy + i * CELL), (ox + BOARD_SIZE, oy + i * CELL), (0, 255, 0), thick)
        cv2.line(out, (ox + i * CELL, oy), (ox + i * CELL, oy + BOARD_SIZE), (0, 255, 0), thick)

    # ตัวเลขบนกระดาน (ชัด + ขอบดำ)
    for i in range(9):
        for j in range(9):
            v = int(board[i, j])
            if v != 0:
                color = (255, 255, 255) if fixed_mask[i, j] else (0, 255, 255)
                (tw, th), _ = cv2.getTextSize(str(v), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                tx = ox + j * CELL + (CELL - tw) // 2
                ty = oy + i * CELL + (CELL + th) // 2 - 4
                cv2.putText(out, str(v), (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(out, str(v), (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    # เตือนผิดกติกา (กรอบแดงกระพริบ)
    now = time.time()
    if invalid_until is not None and now < invalid_until:
        cv2.rectangle(out, (ox-4, oy-4), (ox+BOARD_SIZE+4, oy+BOARD_SIZE+4), (0, 0, 255), 6)
        cv2.putText(out, "INVALID MOVE", (ox, oy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(out, "INVALID MOVE", (ox, oy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 1, cv2.LINE_AA)

    return out

def draw_win_banner(img):
    x, y = LEFT, TOP + BOARD_SIZE // 2 - 40
    w, h = 360, 80
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 0), -1)
    cv2.putText(img, "YOU WIN!", (x + 30, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)

# ===== Main =====
def main():
    cap = cv2.VideoCapture(0)
    # ปรับความละเอียดเฟรมจากกล้อง
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # หน้าต่างยืดได้
    cv2.namedWindow("Sudoku AR - Left/Right Hands", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sudoku AR - Left/Right Hands", 1600, 1000)

    detector = HandDetector(detectionCon=0.8, maxHands=2)

    puzzle, full, board, fixed = new_game(target_empties=50)
    selected = (-1, -1)
    win = False

    # สำหรับ gesture debounce (มือขวาใส่เลข)
    last_digit = None
    digit_hold_start = 0.0
    last_commit = 0.0
    invalid_flash_until = None

    while True:
        ok, raw = cap.read()
        if not ok:
            break

        frame = cv2.flip(raw, 1) if MIRROR else raw

        # ตรวจมือ (สูงสุด 2 มือ)
        hands, frame = detector.findHands(frame)

        # จัดกลุ่มมือซ้าย/ขวา (ตาม label)
        left_hand = None
        right_hand = None
        if hands:
            for h in hands:
                htype = h.get("type", "")
                # ถ้าใช้ MIRROR และต้องการสลับ label
                if MIRROR and SWAP_HANDEDNESS_IF_MIRROR:
                    htype = "Left" if htype == "Right" else ("Right" if htype == "Left" else htype)
                if htype == "Left":
                    left_hand = h
                elif htype == "Right":
                    right_hand = h

        # ====== ควบคุมด้วยมือซ้าย (เลือกช่อง + ลบ) ======
        if left_hand:
            lmL = left_hand.get("lmList", [])
            fingersL = detector.fingersUp(left_hand) if lmL else None

            if lmL and len(lmL) >= 9:
                ixL, iyL, _ = lmL[8]  # ปลายนิ้วชี้ซ้าย
                ixL, iyL = int(ixL), int(iyL)

                # เลือกช่อง: นิ้วชี้ซ้ายยก 1 นิ้วเท่านั้น (แพทเทิร์น [*,1,0,0,0])
                if fingersL and fingersL[1] == 1 and sum(fingersL) == 1:
                    if LEFT <= ixL < LEFT + BOARD_SIZE and TOP <= iyL < TOP + BOARD_SIZE:
                        r = int((iyL - TOP) // CELL)
                        c = int((ixL - LEFT) // CELL)
                        selected = (r, c)

                # ลบค่า: กำมือซ้าย (ทุกนิ้ว 0)
                if fingersL and sum(fingersL) == 0:
                    if 0 <= selected[0] < 9 and 0 <= selected[1] < 9 and not fixed[selected] and not win:
                        board[selected] = 0

                # แสดงตำแหน่งปลายนิ้วชี้ซ้าย (จุดสีน้ำเงิน)
                cv2.circle(frame, (ixL, iyL), 8, (255, 0, 0), -1)

        # ====== ควบคุมด้วยมือขวา (ใส่เลขเท่านั้น) ======
        if right_hand and 0 <= selected[0] < 9 and 0 <= selected[1] < 9 and not fixed[selected] and not win:
            lmR = right_hand.get("lmList", [])
            fingersR = detector.fingersUp(right_hand) if lmR else None

            if fingersR is not None:
                digit = classify_digit_right(fingersR)

                now = time.time()
                # ถ้าท่าดิจิตเปลี่ยน เริ่มจับเวลาใหม่
                if digit != last_digit:
                    last_digit = digit
                    digit_hold_start = now

                # ต้องเป็นตัวเลขจริง 1..9, ค้างพอ, และผ่านคูลดาวน์
                if digit and (now - digit_hold_start) >= GESTURE_HOLD and (now - last_commit) >= GESTURE_COOLDOWN:
                    r, c = selected
                    v = int(digit)
                    if STRICT_VALIDATE and not is_safe(board, r, c, v):
                        invalid_flash_until = now + 0.5
                        last_commit = now
                    else:
                        board[r, c] = v
                        invalid_flash_until = None
                        last_commit = now

        # ===== วาด UI =====
        frame = draw_overlay(frame, board, selected, fixed, invalid_until=invalid_flash_until)

        # คำแนะนำ
        cv2.putText(frame, "Left hand: index=select cell, fist=delete. Right hand: hold gesture=1..9. N=new, Q=quit.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, "Left hand: index=select cell, fist=delete. Right hand: hold gesture=1..9. N=new, Q=quit.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

        # ===== Keyboard fallback =====
        k = cv2.waitKey(1) & 0xFF
        if ord('1') <= k <= ord('9'):
            if 0 <= selected[0] < 9 and 0 <= selected[1] < 9 and not fixed[selected] and not win:
                v = k - ord('0')
                r, c = selected
                if STRICT_VALIDATE and not is_safe(board, r, c, v):
                    invalid_flash_until = time.time() + 0.5
                else:
                    board[r, c] = v
                    invalid_flash_until = None
        elif k in (ord('0'), ord(' ')) or k == 8:   # Space/0/Backspace => delete
            if 0 <= selected[0] < 9 and 0 <= selected[1] < 9 and not fixed[selected] and not win:
                board[selected] = 0
        elif k == ord('n'):  # new game
            puzzle, full, board, fixed = new_game(target_empties=50)
            selected = (-1, -1)
            win = False
            invalid_flash_until = None
            last_digit = None
        elif k == ord('q'):
            break

        # ===== Check win =====
        if not win and np.array_equal(board, full):
            win = True
        if win:
            draw_win_banner(frame)

        cv2.imshow("Sudoku AR - Left/Right Hands", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
