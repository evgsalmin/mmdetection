import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_font('DejaVu', '', 'DejaVuSans.ttf')
        self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf')

        # 2. Меняем стандартный шрифт на только что добавленный
        self.font_family = 'DejaVu'

    def header(self):
        """Создаёт шапку для каждой страницы"""
        self.set_font(self.font_family, 'B', 15)
        self.cell(0, 10, 'Сравнение моделей FCOS и YOLOv8', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font(self.font_family, '', 8)
        self.cell(0, 5, f'Дата генерации: {datetime.date.today().strftime("%d.%m.%Y")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """Добавляет номера страниц в подвале"""
        self.set_y(-15)
        self.set_font(self.font_family, 'B', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title):
        """Создаёт заголовок раздела"""
        self.set_font(self.font_family, 'B', 12)
        self.cell(0, 10, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, body):
        """Добавляет основной текст"""
        self.set_font(self.font_family, '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_image_section(self, title, image_path, stats_text):
        """Добавляет секцию с изображением и статистикой"""
        self.add_page()
        self.chapter_title(title)

        # Центрируем изображение на странице
        image_width = 100
        page_width = self.w - 2 * self.l_margin
        x_position = (page_width - image_width) / 2 + self.l_margin
        self.image(image_path, x=x_position, y=None, w=image_width)
        self.ln(5)
        self.set_font(self.font_family, '', 10) 
        self.multi_cell(0, 5, stats_text)