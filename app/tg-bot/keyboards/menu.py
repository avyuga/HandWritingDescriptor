from aiogram import Bot
from aiogram.types import BotCommand


async def default_commands(bot: Bot):
    menu_commands = [
        BotCommand(command="/start", description="Старт бота"),
        BotCommand(command="/help", description="Помощь"),
        BotCommand(command="/clear", description="Очистить историю сообщений"),

    ]

    await bot.set_my_commands(menu_commands)