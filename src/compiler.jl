if occursin("-DEV", string(VERSION)) || occursin("-beta", string(VERSION))
    const CC = Base
else
    const CC = Core.Compiler
end


