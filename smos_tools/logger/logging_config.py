logging_config = {
    'version': 1,
    'disable_existing_loggers': False,  # this stops overriding existing loggers
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(filename)s - %(levelname)s]: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'  # now they don't print as red
        },

    },
    'loggers': {
        # root logger
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}