from . import Module, register

# noboolean: adds the custom 'containers' OLED page (custom-oled/pages/containers.py)
register(Module(
    name="oled",
    peripherals=[
        'oled',
        'oled_sleep',
        'oled_page_mix',
        'oled_page_performance',
        'oled_page_ips',
        'oled_page_disk',
        'oled_page_containers',
    ],
    default_config={
        'oled_enable': True,
        'oled_rotation': 0,
        'oled_sleep_timeout': 10,
        'oled_pages': [
            'mix',
            'performance',
            'ips',
            'disk',
            'containers',
        ],
    },
))
