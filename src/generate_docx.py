import os
import re
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_shading(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_styled_document(md_path, docx_path, figures_dir):
    doc = Document()
    
    # Page setup - 1 inch margins all around
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Set default body font to Calibri
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x2B, 0x23, 0x1D) # Deep Charcoal Black
    
    # Palette colors
    C_FORBIDDEN_RED = RGBColor(0x9B, 0x1D, 0x20)
    C_TEMPLE_BLUE = RGBColor(0x1F, 0x4E, 0x79)
    C_GOLD = RGBColor(0xDD, 0xA1, 0x2A)
    C_GREY = RGBColor(0x55, 0x55, 0x55)
    HEX_HEADER_BG = "1F4E79"
    HEX_ALT_BG = "F7F5F0"
    HEX_BORDER = "CCCCCC"

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_table = False
    table_lines = []
    in_code_block = False
    code_lines = []

    def clean_text(txt):
        # Convert markdown formatting to plain text with math cleaned
        txt = txt.strip()
        txt = re.sub(r'\$\$(.*?)\$\$', r'\1', txt)
        txt = re.sub(r'\$(.*?)\$', r'\1', txt)
        txt = txt.replace(r'\mu\text{g/m}^3', 'µg/m³')
        txt = txt.replace(r'\text{PM}_{2.5}', 'PM2.5')
        txt = txt.replace(r'\text{PM}_{10}', 'PM10')
        txt = txt.replace(r'\text{SO}_2', 'SO2')
        txt = txt.replace(r'\text{NO}_2', 'NO2')
        txt = txt.replace(r'\text{CO}', 'CO')
        txt = txt.replace(r'\text{O}_3', 'O3')
        txt = txt.replace(r'\text{RMSE}', 'RMSE')
        txt = txt.replace(r'\text{MAE}', 'MAE')
        txt = txt.replace(r'R^2', 'R²')
        txt = txt.replace(r'^\circ\text{C}', '°C')
        txt = txt.replace(r'\approx', '≈')
        txt = txt.replace(r'\le', '≤')
        txt = txt.replace(r'\ge', '≥')
        txt = txt.replace(r'\rightarrow', '→')
        txt = txt.replace(r'\beta', 'β')
        txt = txt.replace(r'\Delta', 'Δ')
        txt = txt.replace(r'\times', '×')
        return txt

    def add_styled_table(tbl_lines):
        rows_data = []
        for line in tbl_lines:
            if not line.strip().startswith('|'):
                continue
            cells = [c.strip() for c in line.strip().split('|')[1:-1]]
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                continue
            rows_data.append(cells)
            
        if not rows_data:
            return
            
        num_cols = max(len(r) for r in rows_data)
        table = doc.add_table(rows=len(rows_data), cols=num_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        for r_idx, row in enumerate(rows_data):
            for c_idx in range(num_cols):
                cell = table.cell(r_idx, c_idx)
                val = row[c_idx] if c_idx < len(row) else ''
                clean_val = clean_text(val).replace('**', '')
                cell.text = clean_val
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
                
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = p.runs[0]
                
                if r_idx == 0:
                    set_cell_shading(cell, HEX_HEADER_BG)
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.font.size = Pt(10)
                else:
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(0x2B, 0x23, 0x1D)
                    if r_idx % 2 == 1:
                        set_cell_shading(cell, HEX_ALT_BG)
                    else:
                        set_cell_shading(cell, "FFFFFF")
                        
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_before = Pt(4)
        p_space.paragraph_format.space_after = Pt(4)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle Code Blocks
        if stripped.startswith('```'):
            if in_code_block:
                in_code_block = False
                # Process code block
                block_text = '\n'.join(code_lines)
                # Check if it references a figure image!
                m_fig = re.search(r'File:\s*reports/figures/([a-zA-Z0-9_\.]+\.png)', block_text)
                if m_fig:
                    img_name = m_fig.group(1)
                    img_path = os.path.join(figures_dir, img_name)
                    if os.path.exists(img_path):
                        p_img = doc.add_paragraph()
                        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        p_img.paragraph_format.space_before = Pt(8)
                        p_img.paragraph_format.space_after = Pt(2)
                        p_img.add_run().add_picture(img_path, width=Inches(6.2))
                        
                        # Add caption
                        caption_m = re.search(r'FIGURE \d+:([^\n]+)', block_text)
                        caption = caption_m.group(0).strip() if caption_m else img_name
                        p_cap = doc.add_paragraph()
                        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        p_cap.paragraph_format.space_before = Pt(0)
                        p_cap.paragraph_format.space_after = Pt(8)
                        r_cap = p_cap.add_run(caption)
                        r_cap.font.size = Pt(9.5)
                        r_cap.font.italic = True
                        r_cap.font.color.rgb = C_GREY
                else:
                    # Render as styled callout box
                    p_code = doc.add_paragraph()
                    p_code.paragraph_format.space_before = Pt(6)
                    p_code.paragraph_format.space_after = Pt(6)
                    p_code.paragraph_format.left_indent = Inches(0.4)
                    r_code = p_code.add_run(block_text)
                    r_code.font.name = 'Consolas'
                    r_code.font.size = Pt(9)
                    r_code.font.color.rgb = C_TEMPLE_BLUE
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line.rstrip('\r\n'))
            i += 1
            continue

        # Handle Markdown Tables
        if stripped.startswith('|'):
            if not in_table:
                in_table = True
                table_lines = [stripped]
            else:
                table_lines.append(stripped)
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                add_styled_table(table_lines)
                table_lines = []

        # Blank Lines
        if not stripped:
            i += 1
            continue

        # Section Headings
        if stripped.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(clean_text(stripped[2:]))
            run.font.name = 'Calibri'
            run.font.size = Pt(20)
            run.font.bold = True
            run.font.color.rgb = C_FORBIDDEN_RED
        elif stripped.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(clean_text(stripped[3:]))
            run.font.name = 'Calibri'
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = C_TEMPLE_BLUE
        elif stripped.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(clean_text(stripped[4:]))
            run.font.name = 'Calibri'
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = C_FORBIDDEN_RED
        elif stripped.startswith('---'):
            # Horizontal rule divider
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run('_________________________________________________________________________')
            run.font.size = Pt(8)
            run.font.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
        elif stripped.startswith('* ') or stripped.startswith('- '):
            # Bullet list item
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            content = stripped[2:]
            
            # Simple bold parsing for lead phrases
            parts = re.split(r'(\*\*.*?\*\*)', content)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(clean_text(part[2:-2]))
                    run.bold = True
                else:
                    p.add_run(clean_text(part))
        elif re.match(r'^\d+\.\s', stripped):
            # Numbered list item
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            content = re.sub(r'^\d+\.\s', '', stripped)
            
            parts = re.split(r'(\*\*.*?\*\*)', content)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(clean_text(part[2:-2]))
                    run.bold = True
                else:
                    p.add_run(clean_text(part))
        else:
            # Regular paragraph
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            
            parts = re.split(r'(\*\*.*?\*\*)', stripped)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(clean_text(part[2:-2]))
                    run.bold = True
                else:
                    p.add_run(clean_text(part))

        i += 1

    if in_table:
        add_styled_table(table_lines)

    doc.save(docx_path)
    print(f"Successfully generated DOCX report at: {docx_path}")

if __name__ == '__main__':
    md_p = 'ML_Mini_Project/reports/final_project_report.md'
    docx_p = 'ML_Mini_Project/reports/final_project_report.docx'
    fig_d = 'ML_Mini_Project/reports/figures'
    create_styled_document(md_p, docx_p, fig_d)
