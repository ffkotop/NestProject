import { Body, Controller, Post } from '@nestjs/common';
import { CreateUserDto, LoginUserDto} from '../dto/user.dto'
import { UserService } from './user.service'
import { Public } from './userjwt.guard'
@Controller('user')
export class UserController {
    constructor(private readonly userService: UserService) {}
    @Post('createUser')
    @Public()
    async create(@Body() AuthInfo: CreateUserDto) {
        return this.userService.createUser(AuthInfo)
    };
    @Post('loginUser')
    @Public()
    async auth(@Body() AuthInfo : LoginUserDto) {
        const user = await this.userService.verifyUser(AuthInfo)
        return this.userService.login(user)

    }

}
