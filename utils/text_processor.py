import re

def clean_continuous_text(text: str) -> str:
    """
    ลบ Newline, Tabs, ช่องว่างที่ซ้ำซ้อน และอักขระควบคุมทั้งหมด
    เพื่อให้ได้ข้อความที่เรียงต่อกันเป็นเนื้อเดียวอย่างสมบูรณ์
    """
    if not text:
        return ""
    
    # 1. แทนที่อักขระควบคุมกลุ่ม \n, \r, \t และอักขระที่ถูก escape (\\n) ด้วยช่องว่าง
    # เราเพิ่มการดักจับ String '\\n' ที่อาจหลุดออกมาจากการประมวลผล JSON
    text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
    
    # 2. ใช้ Regex จัดการ Whitespace ทุกรูปแบบ (\s) และอักขระควบคุม ([\x00-\x1f\x7f-\x9f])
    # วิธีนี้จะล้าง Newline ที่แฝงมาในรูปแบบต่างๆ ได้เด็ดขาด
    text = re.sub(r'[\s\x00-\x1f\x7f-\x9f]+', ' ', text)
    
    # 3. ยุบช่องว่างซ้ำซ้อนให้เหลือเพียง 1 เคาะ และตัดหัวท้าย
    return " ".join(text.split()).strip()