
from funcs import msg_pipeline
import telebot


print("START")


bot = telebot.TeleBot("###")


bot_username = bot.get_me().username



  
    
    


@bot.message_handler(content_types=["text"])
def handle_text(message):

    if f"@{bot_username}" in message.text:
        response  = msg_pipeline("Приветсвую ")
        bot.reply_to(message, response)
        return
    
    # print(message.chat.id)
    text = message.text.strip()
    response = "."
    # if message.reply_to_message:
    response  = msg_pipeline(text)
    bot.send_message(message.chat.id, response)

    
    


bot.polling(none_stop=True, interval=0)