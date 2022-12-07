from PIL import ImageDraw, Image, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os

def Draw_Bboxes(img, bboxes, probs, color=(0,190,0)):
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img[:,:,::-1])       
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'font/arial.ttf'), 15)
    
    for prob, bbox in zip(probs, bboxes):
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
            outline = color,
            width= 2
            )
        text = '%.2f%%'%(prob*100)
        text_size = draw.textsize(text, font)
        
        if bbox[1] - text_size[1] >= 0:
            text_origin = np.array([bbox[0], bbox[1] - text_size[1]])
        else:
            text_origin = np.array([bbox[0], bbox[1] + 1])
        #draw text box and text        
        draw.rectangle(
            [bbox[0], bbox[1], bbox[2], bbox[1]-text_size[1]-6],
            fill = color,
            outline = 'black',
            width = 2
            )
        draw.text([text_origin[0]+5, text_origin[1]-4], text, font= font, fill= 'black')

        # draw.rectangle(
        #     [tuple(text_origin), tuple(text_origin + text_size)],
        #     fill = color, 
        #     outline = 'black'
        #     )
        
        #draw landmark   
        # for p in landmark:
        #     draw.ellipse([
        #         (p[0] - 1.5, p[1] - 1.5),
        #         (p[0] + 1.5, p[1] + 1.5)
        #     ], outline='blue')
    # np_img = np.array(img_copy)
    
    return img_copy

def draw_text(img, text):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img[:,:,::-1])
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'font/arial.ttf'), 15)
    
    draw.text((10,20),text,  font= font, fill = 'black')
    return img_copy

def Draw_face_names(faces, img, confidence = 0.7):
    # img_copy = img.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'font/arial.ttf'), 15)
    margin = 2
    for face in faces:
        if face.top_prediction.confidence <= confidence  :
            text = "Unknown"
            text_size = font.getsize(text)

        else:
            text = "%s:%.2f%%" % (face.top_prediction.label.title(), face.top_prediction.confidence * 100)
            text_size = font.getsize(text)
        
        draw.rectangle(
            (
                (int(face.bb.left), int(face.bb.top)),
                (int(face.bb.right), int(face.bb.bottom))
            ),
            outline=(0,200,0),
            width=2
        )
        draw.rectangle(
            (
                (int(face.bb.left), int(face.bb.top)),
                # (int(face.bb.left + text_size[0] + margin), int(face.bb.top) - text_size[1] + 3 * margin)
                (int(face.bb.left + text_size[0] + margin + 5), int(face.bb.top) - 18)
            ),
            fill = (0, 200, 0),
            outline = 'black',
            width = 2
        )
        draw.text((int(face.bb.left) + 5, int(face.bb.top) - 16), text, fill=(0, 0, 0), font=font)

def Plot(figsize= (25,25), **imgs):

    if len(imgs)> 1:
        _, axes = plt.subplots(1, len(imgs), figsize = figsize)
        axes = axes.flatten()
        
        for (name,img), ax in zip(imgs.items(), axes):
            if img.ndim ==2:
                ax.imshow(img,cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(name)
        plt.show()

    else:
        img = list(imgs.values())[0]
        if img.ndim == 2:
            plt.imshow(img, cmap= 'gray'); plt.show()
        else:
            plt.imshow(img); plt.show()




