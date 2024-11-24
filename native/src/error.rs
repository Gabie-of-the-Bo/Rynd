#[macro_export]
macro_rules! rynd_error {
    ($pat: expr $( , $more: expr)*) => {
        {
            use colored::Colorize;

            eprintln!(
                "[{}] {}",
                "Error".red(),
                format!($pat, $($more,)*)
            );
    
            std::process::exit(1);
        }
    };
}