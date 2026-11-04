use std::io::IsTerminal;

#[derive(Clone, Copy)]
pub(crate) enum Color {
    Cyan,
    Green,
    Yellow,
    Red,
}

impl Color {
    fn ansi_code(self) -> &'static str {
        match self {
            Self::Cyan => "36",
            Self::Green => "32",
            Self::Yellow => "33",
            Self::Red => "31",
        }
    }
}

pub(crate) fn stdout(text: &str, color: Color) -> String {
    colorize(
        text,
        color,
        std::io::stdout().is_terminal(),
        std::env::var_os("NO_COLOR").is_some(),
    )
}

pub(crate) fn stderr(text: &str, color: Color) -> String {
    colorize(
        text,
        color,
        std::io::stderr().is_terminal(),
        std::env::var_os("NO_COLOR").is_some(),
    )
}

fn colorize(text: &str, color: Color, is_terminal: bool, no_color: bool) -> String {
    if is_terminal && !no_color {
        format!("\x1b[{}m{text}\x1b[0m", color.ansi_code())
    } else {
        text.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_color_only_when_terminal_color_is_enabled() {
        assert_eq!(
            colorize("ready", Color::Green, true, false),
            "\x1b[32mready\x1b[0m"
        );
        assert_eq!(colorize("ready", Color::Green, false, false), "ready");
        assert_eq!(colorize("ready", Color::Green, true, true), "ready");
    }
}
