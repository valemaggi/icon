% depression_rules.pl

depression('1') :- academic_pressure('alta').
depression('1') :- financial_stress('5.0').
depression('1') :- have_you_ever_had_suicidal_thoughts_('yes').
depression('1') :- study_satisfaction('bassa'), dietary_habits('unhealthy').
depression('1') :- age('universitario'), dietary_habits('unhealthy').
depression('1') :- sleep_duration('less_than_5_hours'), dietary_habits('unhealthy').

depression('0') :- \+ depression('1').
