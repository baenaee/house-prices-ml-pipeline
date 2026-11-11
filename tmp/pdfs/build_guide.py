from pathlib import Path
import re, csv, json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/house_prices_feature_guide.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
for name, file in [('Regular','calibri.ttf'),('Bold','calibrib.ttf'),('Light','calibril.ttf'),('Italic','calibrii.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Regular',normal='Regular',bold='Bold',italic='Italic',boldItalic='Bold')
raw = (ROOT/'data_description.txt').read_text()
features = {}
for match in re.finditer(r'^([A-Za-z0-9]+):\s*([^\n]+)(.*?)(?=^[A-Za-z0-9]+:|\Z)', raw, re.M|re.S):
    key, desc, rest = match.groups()
    opts=[]
    for line in rest.splitlines():
        if not line.strip(): continue
        parts=re.split(r'\t+| {2,}',line.strip(),maxsplit=1)
        assert len(parts)==2, (key,line)
        opts.append(tuple(p.strip() for p in parts))
    features[key]=(desc.strip(),opts)

GROUPS=[
('Property identity','Dwelling form, construction history and whole-home ratings.', 'MSSubClass BldgType HouseStyle YearBuilt YearRemodAdd OverallQual OverallCond'),
('Land & access','Lot dimensions, shape, topography, zoning and basic services.', 'MSZoning LotFrontage LotArea Street Alley LotShape LandContour Utilities LotConfig LandSlope'),
('Location & surroundings','Ames neighborhoods and nearby streets, railways or positive features.', 'Neighborhood Condition1 Condition2'),
('Exterior & structure','Roofing, exterior finishes, masonry, foundation and exterior condition.', 'RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType MasVnrArea ExterQual ExterCond Foundation'),
('Basement','Height, condition, exposure, finish types and below-grade floor area.', 'BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinSF1 BsmtFinType2 BsmtFinSF2 BsmtUnfSF TotalBsmtSF'),
('Living space & rooms','Above-grade space, rooms, bathrooms and kitchen quality.', '1stFlrSF 2ndFlrSF LowQualFinSF GrLivArea BsmtFullBath BsmtHalfBath FullBath HalfBath Bedroom Kitchen KitchenQual TotRmsAbvGrd'),
('Comfort & function','Heating, cooling, electrical service, functionality and fireplaces.', 'Heating HeatingQC CentralAir Electrical Functional Fireplaces FireplaceQu'),
('Garage & parking','Garage location, age, capacity, finish and quality; driveway paving.', 'GarageType GarageYrBlt GarageFinish GarageCars GarageArea GarageQual GarageCond PavedDrive'),
('Outdoor & extra features','Decks, porches, pools, fences and miscellaneous improvements.', 'WoodDeckSF OpenPorchSF EnclosedPorch 3SsnPorch ScreenPorch PoolArea PoolQC Fence MiscFeature MiscVal'),
('Sale details','Sale timing, transaction type and circumstances of the sale.', 'MoSold YrSold SaleType SaleCondition'),
]
ordered=[k for _,_,keys in GROUPS for k in keys.split()]
assert len(ordered)==len(set(ordered))==len(features)==79
assert set(ordered)==set(features)

W,H=595.276,841.89
M=40
CW=(W-2*M-22)/2
NAVY=HexColor('#142D3B'); TEAL=HexColor('#007E80'); INK=HexColor('#253D49')
MUTED=HexColor('#607784'); PAPER=HexColor('#F5F7F7'); LINE=HexColor('#DCE5E7')
GOLD=HexColor('#E5B366')
styles={
'body':ParagraphStyle('body',fontName='Regular',fontSize=10.4,leading=14,textColor=INK),
'desc':ParagraphStyle('desc',fontName='Regular',fontSize=10,leading=13,textColor=INK),
'option':ParagraphStyle('option',fontName='Regular',fontSize=9.2,leading=11.7,textColor=INK),
'code':ParagraphStyle('code',fontName='Bold',fontSize=9.2,leading=11.7,textColor=TEAL),
'small':ParagraphStyle('small',fontName='Regular',fontSize=9,leading=12,textColor=MUTED),
}

def para(text,x,y,w,style='body',draw=True):
    p=Paragraph(text,styles[style]); _,h=p.wrap(w,900)
    if draw:p.drawOn(c,x,y-h)
    return h

def txt(text,x,y,size=10,font='Regular',color=INK):
    c.setFillColor(color);c.setFont(font,size);c.drawString(x,y,text)

def base(page,label):
    c.setFillColor(white);c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(TEAL);c.rect(M,H-30,26,3,fill=1,stroke=0)
    txt('HOUSE PRICES  /  FEATURE GUIDE',M+36,H-31,8,'Bold',MUTED)
    c.setStrokeColor(LINE);c.line(M,41,W-M,41)
    txt(label.upper(),M,25,8,'Bold',MUTED)
    c.setFont('Regular',9);c.setFillColor(MUTED);c.drawRightString(W-M,25,f'{page:02d}')

aliases={'Bedroom':'BedroomAbvGr','Kitchen':'KitchenAbvGr'}
def display(k):return aliases.get(k,k)
def card(k,x,y,draw=True):
    desc,opts=features[k]
    inn=CW-20
    dh=para(escape(desc),0,0,inn,'desc',False)
    alias=12 if k in aliases else 0
    rowdata=[]
    for code,meaning in opts:
        rh=max(para(escape(code),0,0,60,'code',False),para(escape(meaning),0,0,inn-69,'option',False))+6
        rowdata.append((code,meaning,rh))
    ht=12+18+alias+dh+8+sum(r[2] for r in rowdata)+6
    if not draw:return ht
    c.setFillColor(PAPER);c.roundRect(x,y-ht,CW,ht,6,fill=1,stroke=0)
    txt(display(k),x+10,y-23,13,'Bold',NAVY)
    cy=y-31
    if alias:
        txt('Source name: '+k,x+10,cy-7,8,'Italic',MUTED);cy-=12
    para(escape(desc),x+10,cy,inn,'desc');cy-=dh+8
    for i,(code,meaning,rh) in enumerate(rowdata):
        if i%2==0:
            c.setFillColor(white);c.rect(x+7,cy-rh,CW-14,rh,fill=1,stroke=0)
        para(escape(code),x+10,cy-3,60,'code')
        para(escape(meaning),x+79,cy-3,inn-69,'option')
        cy-=rh
    return ht

# Lay out all feature cards before drawing, so contents and index are accurate.
pages=[]; feature_pages={}; group_pages=[]
for gi,(title,summary,keys) in enumerate(GROUPS):
    group_pages.append(3+len(pages))
    cards=[]; ys=[690,690]; continued=False
    for k in keys.split():
        ht=card(k,0,0,False)
        assert ht<630,(k,ht)
        col=0 if ys[0]>=ys[1] else 1
        if ys[col]-ht<59 or (cards and k in {'YearBuilt','LandContour'}):
            pages.append((gi,continued,cards));cards=[];ys=[690,690];continued=True;col=0
        feature_pages[k]=3+len(pages)
        cards.append((k,M+col*(CW+22),ys[col],ht))
        ys[col]-=ht+13
    pages.append((gi,continued,cards))

c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('House Prices | A field guide to the features')
c.setAuthor('Project reference guide')
c.setSubject('Complete grouped reference to the 79 features in data_description.txt')

# Cover
c.setFillColor(NAVY);c.rect(0,0,W,H,fill=1,stroke=0)
c.setFillColor(TEAL);c.rect(0,H-16,W,16,fill=1,stroke=0)
txt('KAGGLE  /  ADVANCED REGRESSION',M,766,10,'Bold',HexColor('#A7DADA'))
txt('House',M,650,66,'Light',white)
txt('Prices',M,581,66,'Bold',white)
txt('A field guide to the features',M,530,23,'Light',white)
c.setStrokeColor(HexColor('#496571'));c.line(M,495,W-M,495)
coverstyle=ParagraphStyle('cover',fontName='Regular',fontSize=14,leading=20,textColor=HexColor('#D3E1E5'))
p=Paragraph('Every definition. Every category code.<br/>A clear, organized reference to the characteristics<br/>of homes in Ames, Iowa.',coverstyle);p.wrap(W-2*M,150);p.drawOn(c,M,408)
# Architectural line illustration.
c.setStrokeColor(HexColor('#5CA7AA'));c.setLineWidth(1.5)
path=c.beginPath();path.moveTo(330,176);path.lineTo(330,274);path.lineTo(418,342);path.lineTo(507,274);path.lineTo(507,176);path.close();c.drawPath(path)
c.line(312,274,418,357);c.line(418,357,525,274)
c.rect(391,176,48,77,stroke=1,fill=0);c.rect(349,270,31,30,stroke=1,fill=0);c.rect(454,225,30,30,stroke=1,fill=0)
c.line(310,176,530,176)
txt('79',M,286,39,'Bold',GOLD);txt('FEATURE DEFINITIONS',M,266,9,'Bold',white)
txt('10',M,219,39,'Bold',GOLD);txt('THEMED SECTIONS',M,199,9,'Bold',white)
txt('Based on the project data_description.txt',M,85,10,'Regular',HexColor('#D3E1E5'))
txt('Includes a linked contents page and alphabetical feature index',M,66,9,'Regular',HexColor('#A7BEC7'))
c.showPage()

# Reader orientation and linked contents.
base(2,'Start here')
txt('A clearer way in',M,748,30,'Bold',NAVY)
para('Browse by topic, or jump directly to a feature using the alphabetical index. Field names and category codes are retained for easy lookup.',M,724,W-2*M)
txt('CONTENTS',M,665,9,'Bold',TEAL)
y=641
for gi,(title,summary,keys) in enumerate(GROUPS):
    txt(f'{gi+1:02d}',M,y,10,'Bold',TEAL)
    txt(title,M+29,y,12,'Bold',NAVY)
    c.setFillColor(MUTED);c.setFont('Regular',10);c.drawRightString(W-M,y,f'{group_pages[gi]:02d}')
    c.linkRect('',f'group{gi}',(M,y-7,W-M,y+15),relative=0,thickness=0)
    y-=29
idx_page=3+len(pages)
txt('A-Z',M,y,10,'Bold',TEAL);txt('Alphabetical feature index',M+29,y,12,'Bold',NAVY)
c.setFont('Regular',10);c.drawRightString(W-M,y,str(idx_page))
c.linkRect('','index',(M,y-7,W-M,y+15),relative=0,thickness=0)
c.setStrokeColor(LINE);c.line(M,y-22,W-M,y-22)
y-=45
for title,body in [
('Read codes literally','Codes are labels, even when they look numeric. MSSubClass identifies dwelling types; OverallQual and OverallCond use ordered 1-10 ratings.'),
('Interpret absence by feature','NA explicitly means absence for several features, such as no basement or no garage. Treat those documented meanings separately from unknown or missing measurements.'),
('Match the project columns','The source calls two fields Bedroom and Kitchen; train.csv names them BedroomAbvGr and KitchenAbvGr. This guide shows the dataset names and the original names together.'),
('Scope & fidelity','All 79 source definitions and their listed codes are included. Id and the SalePrice target appear in train.csv but are not defined in the text file. Source wording and code spellings are retained; check actual dataset values before encoding.')]:
    txt(title,M,y,11,'Bold',NAVY);y-=8
    h=para(escape(body),M,y,W-2*M,'small');y-=h+19
assert y>45,y
c.showPage()

for pi,(gi,continued,cards) in enumerate(pages):
    page=3+pi;title,summary,keys=GROUPS[gi]
    base(page,title)
    if not continued:
        c.bookmarkPage('group'+str(gi));c.addOutlineEntry(title,'group'+str(gi),level=0)
    txt(f'{gi+1:02d}  /  {len(keys.split())} FEATURES'+('  /  CONTINUED' if continued else ''),M,778,9,'Bold',TEAL)
    txt(title,M,743,27,'Bold',NAVY)
    para(escape(summary),M,723,W-2*M,'small')
    for k,x,y,ht in cards:
        c.bookmarkHorizontalAbsolute('f_'+k,y)
        card(k,x,y)
    c.showPage()

base(idx_page,'Feature index')
c.bookmarkPage('index');c.addOutlineEntry('Alphabetical feature index','index',level=0)
txt('Find a feature',M,748,30,'Bold',NAVY)
para('Dataset column names, alphabetically arranged. Click a name to jump to its definition.',M,724,W-2*M,'small')
items=sorted(features,key=lambda k:display(k).lower())
for i,k in enumerate(items):
    col=i//40;row=i%40;x=M+col*(CW+22);y=682-row*15
    txt(display(k),x,y,9.5,'Regular',INK)
    c.setFont('Bold',9);c.setFillColor(TEAL);c.drawRightString(x+CW,y,str(feature_pages[k]))
    c.linkRect('','f_'+k,(x,y-3,x+CW,y+11),relative=0,thickness=0)
c.save()

reader=PdfReader(str(OUT)); extracted='\n'.join(p.extract_text() for p in reader.pages)
norm=lambda s:re.sub(r'\s+','',s)
flat=norm(extracted)
for k,(desc,opts) in features.items():
    assert norm(display(k)) in flat,k
    assert norm(desc) in flat,(k,'description missing')
    for code,meaning in opts:
        assert norm(code) in flat,(k,code)
        assert norm(meaning) in flat,(k,meaning)
report={'pages':len(reader.pages),'features':len(features),'category_definitions':sum(len(v[1]) for v in features.values()),'group_pages':group_pages,'output':str(OUT),'all_source_definitions_verified':True}
(ROOT/'tmp/pdfs/validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
