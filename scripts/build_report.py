"""Build the complete four-task Markdown and PDF report."""
from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdfcanvas
from report_content import REPORT_PAGES

ROOT=Path(__file__).resolve().parents[1]
FONT=Path('/usr/share/fonts/truetype/dejavu')
# Fonts are bundled in the submission for portable reproduction.
if (ROOT/'assets/DejaVuSans.ttf').exists():FONT=ROOT/'assets'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:
 pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
INK=HexColor('#142D43');TEAL=HexColor('#007F82');GRAY=HexColor('#5D6E7B');LIGHT=HexColor('#EDF3F5')
W,H=A4;CONTENT=W-88
styles={
 'p':ParagraphStyle('body',fontName='Body',fontSize=9.2,leading=13.4,textColor=INK,spaceAfter=8),
 'title':ParagraphStyle('title',fontName='Bold',fontSize=23,leading=28,textColor=INK,spaceAfter=7),
 'subtitle':ParagraphStyle('subtitle',fontName='Body',fontSize=10,leading=14,textColor=GRAY,spaceAfter=18),
 'h':ParagraphStyle('h',fontName='Bold',fontSize=11.5,leading=15,textColor=TEAL,spaceBefore=6,spaceAfter=8,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Body',fontSize=8,leading=11,textColor=GRAY,spaceBefore=5,spaceAfter=10),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.5,leading=11.6,textColor=INK),
 'head':ParagraphStyle('head',fontName='Bold',fontSize=8.6,leading=11.6,textColor=white),
 'kpi':ParagraphStyle('kpi',fontName='Bold',fontSize=22,leading=25,textColor=TEAL,alignment=1),
 'kpilabel':ParagraphStyle('kpilabel',fontName='Body',fontSize=8.5,leading=11,textColor=GRAY,alignment=1)
}
def markup(text):
 text=html.escape(text)
 text=re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',r'<a href="\2" color="#007F82">\1</a>',text)
 text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
 text=re.sub(r'`([^`]+)`',r'<font size="8.6">\1</font>',text)
 return text

def para(text,style='p'):return Paragraph(markup(text),styles[style])
def footer(canvas,doc):
 canvas.saveState();canvas.setFillColor(GRAY);canvas.setFont('Body',7.5)
 canvas.drawString(44,H-27,'REACTION / BEHAVIORAL RESPONSE PREDICTION')
 canvas.setStrokeColor(HexColor('#D8E2E8'));canvas.line(44,36,W-44,36)
 canvas.drawString(44,23,'Final submission  |  1 October 2026')
 canvas.restoreState()

class NumberedCanvas(pdfcanvas.Canvas):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.saved_pages=[]
 def showPage(self):
  self.saved_pages.append(dict(self.__dict__));self._startPage()
 def save(self):
  total=len(self.saved_pages)
  for state in self.saved_pages:
   self.__dict__.update(state)
   self.setFillColor(GRAY);self.setFont('Body',7.5)
   self.drawRightString(W-44,23,f'{self._pageNumber} / {total}')
   super().showPage()
  super().save()

def render_pages(pages):
 story=[];md=[]
 for i,page in enumerate(pages):
  if i:story.append(PageBreak());md.extend(['','---',''])
  story.extend([para(page['title'],'title'),para(page['subtitle'],'subtitle')])
  md.extend([('# ' if i==0 else '## ')+page['title'],'',page['subtitle'],''])
  for b in page['blocks']:
   kind=b['type']
   if kind in ['p','h']:
    story.append(para(b['text'],kind));md.extend([('### ' if kind=='h' else '')+b['text'],''])
   elif kind=='callout':
    t=Table([[para('**'+b['text']+'**')]],colWidths=[CONTENT])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,HexColor('#CBDADC')),
     ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    story.extend([t,Spacer(1,12)]);md.extend(['> **'+b['text']+'**',''])
   elif kind=='table':
    cells=[[para(x,'head') for x in b['headers']]]+[[para(str(x),'cell') for x in row] for row in b['rows']]
    fractions=b.get('widths') or [1/len(b['headers'])]*len(b['headers'])
    t=Table(cells,colWidths=[CONTENT*f for f in fractions],hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
     ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,LIGHT]),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
     ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([t,Spacer(1,12)])
    md.extend(['| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*len(b['headers']))+' |'])
    md.extend(['| '+' | '.join(str(x) for x in row)+' |' for row in b['rows']]);md.append('')
   elif kind=='image':
    relative=Path(b.get('directory', 'figures'))/f"{b['name']}.png"
    path=ROOT/relative
    im=Image(str(path));ratio=im.imageWidth/im.imageHeight
    height=min(b['height'],CONTENT/ratio);im.drawHeight=height;im.drawWidth=height*ratio;im.hAlign='CENTER'
    story.append(KeepTogether([im,para(b['caption'],'caption')]))
    md.extend([f"![{b['caption']}]({relative.as_posix()})",'',b['caption'],''])
   elif kind=='kpis':
    cells=[[para(v,'kpi') for v,label in b['items']],[para(label,'kpilabel') for v,label in b['items']]]
    t=Table(cells,colWidths=[CONTENT/len(b['items'])]*len(b['items']))
    t.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,12)])
    md.extend([' | '.join(f'**{v}** {label}' for v,label in b['items']),''])
 return story, md

story, md = render_pages(REPORT_PAGES)
(ROOT/'report.md').write_text('\n'.join(md))
doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=49,bottomMargin=49,
 title='ReAction - Final submission',author='ReAction submission',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer,canvasmaker=NumberedCanvas)
print('Built report.md and report.pdf')
