import re

def clean_continuous_text(text: str) -> str:
    """
    ลบ Newline, Tabs, และช่องว่างที่ซ้ำซ้อนทั้งหมด 
    ให้เหลือเพียงข้อความที่เรียงต่อกันด้วยช่องว่างเดียว
    """
    if not text:
        return ""
    # แทนที่ Newline (\n), Carriage Return (\r), Tab (\t) ด้วยช่องว่าง
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # ยุบช่องว่างที่ซ้ำกันให้เหลือเพียง 1 เคาะ และตัดช่องว่างหัวท้ายออก
    return " ".join(text.split()).strip()