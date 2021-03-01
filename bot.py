import discord
import random
import cv2
import io
import time
import re
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
        config['chance'] = yml['Chance']
        config['min'] = yml['Minimum Word Length']
        config['del'] = True if 'true' in str(yml['Allow Deletion']).lower() else False
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

    # parse discord message
    content = re.sub(r'<.*?>|\'.*?\'|__.*?__|\*.*?\*|>.*?\n|{.*?}', '', message.content, re.S | re.M)

    # detect all reactions
    found_nouns = list()

    for nouns in nlp_analysis.get_noun_phrases(content):
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
    img = img_text.cv_img_contrast(img, 1.2, -60) # Enhance contrast
    img = img_text.demo_text(img, adj_only, size=20/(len(adj_only)+1)+0.5) # Add text
    img = img_text.resize(img, px=600) # Decrease size

    cv2.imwrite('yeet.png', img)

    file = discord.File('yeet.png')
    embed = discord.Embed(title='You probably wanted...', description=f'{str(noun_phrase).title()}\nReact with ❌ to delete' if config['del'] else f'{str(noun_phrase).title()}')
    embed.set_image(url='attachment://yeet.png')

    # resize to 1 px and get the color for the discord embed
    col = img_text.resize(img, px=1)[0,0][0:3]
    embed.colour = discord.Colour.from_rgb(int(col[0]), int(col[1]), int(col[2]))

    # send the picture!
    await message.channel.send(file = file, embed=embed)

@bot.event
async def on_reaction_add(reaction, user): # Don't use raw here, as we want msg to expire
    if not config['del']: return # must have enabled deletion
    if user.bot: return # reactor must not be a bot
    if reaction.message.author.id != bot.user.id: return # must be on your message
    # Make sure this message is an image reaction
    if len(reaction.message.embeds) == 0: return # must have an embed
    if not 'yeet.png' in repr(reaction.message.embeds[0].image): return # make sure 

    # if the reaction is an X, delete the message
    if '❌' in reaction.emoji:
        await reaction.message.delete()

bot.run(config['token'])