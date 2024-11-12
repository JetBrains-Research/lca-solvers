class ReprMixin:
    _init_args = None
    _init_kwargs = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def wrapped_init(self, *init_args, **init_kwargs) -> None:
            if cls == type(self):
                self._init_args = init_args
                self._init_kwargs = init_kwargs
            original_init(self, *init_args, **init_kwargs)

        cls.__init__ = wrapped_init

    def __repr__(self) -> str:
        args_str = ', '.join(
            [
                f"'{arg}'" if isinstance(arg, str) else repr(arg)
                for arg in self._init_args
            ] + [
                f"{key}='{value}'" if isinstance(value, str) else f'{key}={value!r}'
                for key, value in self._init_kwargs.items()
            ])
        return f'{type(self).__name__}({args_str})'
