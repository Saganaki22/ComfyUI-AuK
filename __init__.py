from .nodes import AuKExtension


async def comfy_entrypoint():
    return AuKExtension()
