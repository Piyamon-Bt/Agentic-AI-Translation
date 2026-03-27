import re
import jieba
import opencc
from g2pM import G2pM
from typing import List, Dict

# โหลด Models ครั้งเดียวที่ระดับ Global เพื่อประหยัด Memory และเวลา
_g2p = G2pM()
_trad_to_simp = opencc.OpenCC("t2s")  # แปลงตัวเต็มเป็นตัวย่อเพื่อให้ g2pM แม่นยำที่สุด

# ---------- Configuration & Constants ----------
VOWELS = 'aeiouvüAEIOUVÜ'
TONE_MARKS = {
    'a': ['ā','á','ǎ','à'], 'e': ['ē','é','ě','è'], 'i': ['ī','í','ǐ','ì'],
    'o': ['ō','ó','ǒ','ò'], 'u': ['ū','ú','ǔ','ù'], 'v': ['ǖ','ǘ','ǚ','ǜ'], 'ü': ['ǖ','ǘ','ǚ','ǜ'],
    'A': ['Ā','Á','Ǎ','À'], 'E': ['Ē','É','Ě','È'], 'I': ['Ī','Í','Ǐ','Ì'],
    'O': ['Ō','Ó','Ǒ','Ò'], 'U': ['Ū','Ú','Ǔ','Ù'], 'V': ['Ǖ','Ǘ','Ǚ','Ǜ'], 'Ü': ['Ǖ','Ǘ','Ǚ','Ǜ'],
}
PRIORITY = ['a','e','o']
NEUTRAL_SET = set(list("的地得了吗呢吧啊着过過們们子頭头"))
_HAN_RE = re.compile(r'[\u3400-\u9FFF\uF900-\uFAFF]')

# ---------- Helper Functions ----------
def _is_han(ch: str) -> bool:
    """เช็คว่าเป็นตัวอักษรจีนหรือไม่"""
    return bool(_HAN_RE.match(ch))

def _strip_tone(s: str) -> str:
    """ตัดตัวเลขโทนออกและจัดการสระ ü"""
    if not s: return s
    base = s[:-1] if s[-1].isdigit() else s
    base = (base.replace('u:', 'ü').replace('U:', 'Ü')
                 .replace('v', 'ü').replace('V', 'Ü'))
    return base

def _choose_tone_index(base: str) -> int:
    """เลือกตำแหน่งที่จะวางเครื่องหมายวรรณยุกต์ตามกฎสระ"""
    low = base.lower()
    if 'iu' in low: return low.index('iu') + 1
    if 'ui' in low: return low.index('ui')
    for p in PRIORITY:
        pos = low.find(p)
        if pos != -1: return pos
    for i, ch in enumerate(base):
        if ch in VOWELS: return i
    return -1

def number_to_mark(syl: str) -> str:
    """เปลี่ยนจาก 'wo3' เป็น 'wǒ'"""
    if not syl: return syl
    tone = syl[-1]
    base = _strip_tone(syl)
    if tone not in '1234': return base
    idx = _choose_tone_index(base)
    if idx == -1: return syl
    vowel = base[idx]
    table = TONE_MARKS.get(vowel)
    if not table: return syl
    marked = table[int(tone)-1]
    return base[:idx] + marked + base[idx+1:]

# ---------- กฎ Sandhi (การเปลี่ยนเสียง) ----------
def apply_extended_sandhi(text_han_only: str, syllables: List[str]) -> List[str]:
    """จัดการกฎ 3+3 -> 2+3 และกฎของ 不, 一"""
    out = syllables[:]
    n = len(out)

    # 1. กฎ 3+3 -> 2+3 (เช่น nǐ hǎo -> ní hǎo)
    for i in range(n - 1):
        if out[i].endswith('3') and out[i+1].endswith('3'):
            out[i] = _strip_tone(out[i]) + '2'

    # 2. กฎของ 不 (bù) และ 一 (yī) และ 轻声 (Neutral tone)
    # หมายเหตุ: text_han_only ต้องมีความยาวเท่ากับ syllables
    for k in range(n):
        ch = text_han_only[k]
        cur = out[k]
        # ดูเสียงของพยางค์ถัดไป
        nxt_tone = out[k+1][-1] if k+1 < n and out[k+1][-1].isdigit() else None
        
        if ch == '不' and nxt_tone == '4':
            out[k] = 'bu2'
        elif ch == '一':
            if nxt_tone == '4': out[k] = 'yi2'
            elif nxt_tone in ('1', '2', '3'): out[k] = 'yi4'
        elif ch in NEUTRAL_SET:
            out[k] = _strip_tone(cur) + '5'
            
    return out

# ---------- Main Conversion Function ----------
def convert_pinyin_both(text: str, use_sandhi: bool = True) -> Dict:
    """
    ฟังก์ชันหลักในการแปลงภาษาจีน (Simplified/Traditional) เป็นพินอิน
    ที่รักษาโครงสร้างเครื่องหมายวรรคตอนและช่องว่างอย่างถูกต้อง
    """
    if not text:
        return {"pinyin_marks": ""}

    # 1. Normalize: แปลงตัวเต็มเป็นตัวย่อ และล้างช่องว่างส่วนเกิน
    simplified_text = _trad_to_simp.convert(text)
    
    # 2. แยกเฉพาะตัวจีนออกมาเพื่อส่งให้ g2pM (ป้องกันเครื่องหมายวรรคตอนทำ Index เลื่อน)
    han_only = "".join([ch for ch in simplified_text if _is_han(ch)])
    if not han_only:
        return {"pinyin_marks": text} # ถ้าไม่มีตัวจีนเลย คืนค่าเดิม

    # 3. แปลงเป็นตัวเลขโทน และใช้กฎ Sandhi
    syl_num = _g2p(han_only, tone=True)
    if use_sandhi:
        syl_num = apply_extended_sandhi(han_only, syl_num)

    # 4. แปลงตัวเลขเป็นเครื่องหมายโทน (Mark)
    syl_mark = [number_to_mark(s) for s in syl_num]

    # 5. ตัดคำจากข้อความต้นฉบับเพื่อรักษาเครื่องหมายวรรคตอน
    words = jieba.lcut(simplified_text)
    
    idx = 0
    final_parts = []
    
    for w in words:
        # นับจำนวนตัวจีนใน "คำ" นี้
        han_chars_in_word = [ch for ch in w if _is_han(ch)]
        num_han = len(han_chars_in_word)
        
        if num_han > 0:
            # ดึงพยางค์พินอินตามจำนวนตัวจีนในคำนั้นมาเชื่อมกัน
            # เช่น 'nǐ' + 'hǎo' -> 'nǐhǎo'
            pinyin_word = "".join(syl_mark[idx : idx + num_han])
            final_parts.append(pinyin_word)
            idx += num_han
        else:
            # ถ้าไม่ใช่ตัวจีน (เครื่องหมายวรรคตอน/ช่องว่าง) ให้ใส่ลงไปตรงๆ
            item = w.strip()
            if item:
                final_parts.append(item)

    # 6. Post-processing: จัดการช่องว่างและตัวพิมพ์ใหญ่
    # รวมร่างด้วยช่องว่าง
    result = " ".join(final_parts)
    
    # ลบช่องว่างหน้าเครื่องหมายวรรคตอน (เช่น 'rúguǒ ,' -> 'rúguǒ,')
    result = re.sub(r'\s+([,。，？！、.；：])', r'\1', result)
    
    # ยุบช่องว่างที่ซ้ำซ้อน
    result = " ".join(result.split())

    # ทำให้ตัวอักษรแรกของประโยคเป็นตัวพิมพ์ใหญ่ (Capitalize)
    if result:
        result = result[0].upper() + result[1:]

    return {
        "pinyin_marks": result
    }