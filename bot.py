import discord
import random
import cv2
import io
import time
import nlp_analysis
import img_text

import yaml
try: from yaml import CLoader as Loader
except ImportError: from yaml import Loader

# load the config
config = dict()
with open('./config.yml') as file:
    yml = yaml.load(file.read(), Loader=Loader)
    try:
        config['token'] = yml['Token']
        config['key'] = yml['Pixabay Key']
        config['min'] = yml['Minimum Word Length']
        config['chance'] = yml['Chance']
    except (KeyError, ValueError): 
        print('Error in config')
        quit(1)
    assert '<TOKEN>' not in repr(config), 'Please add your token to the config!'

# setting up the bot, with its discritpion etc.
bot = discord.Client() # use Client, as we don't need full bot functionality

@bot.event
async def on_ready():
    print('\n\nBooted!\n\n')

# for every message it does these checks
@bot.event
async def on_message(message):
    if random.random() > config['chance']: return

    # detect all reactions
    found_nouns = list()

    for nouns in nlp_analysis.get_noun_phrases(message.content):
        if len(nouns) < config['min']: continue
        found_nouns.append(nouns)
    
    # print nouns found
    if len(found_nouns) > 0: print('Found {0} noun phrases in message "{1}": {2}'.format(len(found_nouns), str(message.content).replace('\n',' '), found_nouns))
    else: return

    # Remove all with only 1 word
    found_nouns = list(filter(lambda x: str(x).count(' ') >= 1, found_nouns))
    if len(found_nouns) <= 0: return

    # Pick a random noun phrase
    noun_phrase = random.choice(found_nouns)
    adj_only, noun_only = noun_phrase[:noun_phrase.rfind(' ')].strip(), noun_phrase[noun_phrase.rfind(' '):].strip()
    print(f'Noun phrase chosen: ({adj_only}) {noun_only}')

    # generate the image
    img = img_text.cv_img_from_url(img_text.img_url_from_query(noun_only, config['key']))
    img = img_text.cv_img_contrast(img, 1.2, -60) # ENHANCEEEE
    
    if False:
        adjs = adj_only.split(' ')
        for i in range(len(adjs)-1, -1, -1):
            img = img_text.demo_text(img, adjs[i], size=20/(len(adjs[i])+1)+0.5, n_start=i*2)
    else:
        img = img_text.demo_text(img, adj_only, size=20/(len(adj_only)+1)+0.5, n_start=i*2)
    img = img_text.resize(img, px=600)
    
    # embed the image (https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/)
    #success, img_buf = cv2.imencode('.jpg', img)
    #io_buf = io.BytesIO(img_buf)
    #img_bytes = io_buf.getvalue()
    #assert success

    cv2.imwrite('yeet.png', img)

    file = discord.File('yeet.png')
    embed = discord.Embed(title='You probably wanted...')
    embed.set_image(url='attachment://yeet.png')
    await message.channel.send(file = file, embed=embed)


bot.run(config['token'])