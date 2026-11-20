//! Small ANSI terminal-coloring helpers.
//!
//! Everywhere in the CLI (banners, progress, errors) text is wrapped by
//! [`stdout`] / [`stderr`], which centralize the decision of whether to
//! emit escape sequences:
//! - the output target is not a TTY (pipe, redirection) → no color;
//! - the `NO_COLOR` environment variable is set → no color (following
//!   the no-color.org convention).
//!
//! Business code therefore never inspects the environment itself; it
//! just calls these wrappers.

use std::io::IsTerminal;

/// The four foreground colors used by the CLI (SGR codes in
/// [`Color::ansi_code`]).
#[derive(Clone, Copy)]
pub(crate) enum Color {
    /// Cyan: progress and secondary information.
    Cyan,
    /// Green: success and readiness.
    Green,
    /// Yellow: warnings.
    Yellow,
    /// Red: errors.
    Red,
}

impl Color {
    /// Returns the SGR foreground code (the ANSI escape parameter) for
    /// this color.
    fn ansi_code(self) -> &'static str {
        match self {
            Self::Cyan => "36",
            Self::Green => "32",
            Self::Yellow => "33",
            Self::Red => "31",
        }
    }
}

/// Colorizes text destined for stdout (returned unchanged when the target
/// is not a TTY or `NO_COLOR` is set).
pub(crate) fn stdout(text: &str, color: Color) -> String {
    colorize(
        text,
        color,
        std::io::stdout().is_terminal(),
        std::env::var_os("NO_COLOR").is_some(),
    )
}

/// Colorizes text destined for stderr; same rules as [`stdout`].
pub(crate) fn stderr(text: &str, color: Color) -> String {
    colorize(
        text,
        color,
        std::io::stderr().is_terminal(),
        std::env::var_os("NO_COLOR").is_some(),
    )
}

/// The coloring core: wraps text in `\x1b[..m ... \x1b[0m` only when the
/// target is a terminal AND color has not been disabled.
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
